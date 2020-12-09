import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import scipy.sparse
import pyamg
from sys import exit as e
from torch.nn import functional as F
from torch import nn
from skimage import exposure
from skimage.exposure import match_histograms
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()
from modules.util import imshow, show
from modules.poisson import solve_poisson


# https://github.com/fbessho/PyPoi
def blend(img_target, img_source, img_mask, offset=(0, 0)):
  new_mask= 1 - img_mask
  copy_img = cv2.bitwise_and(img_source, img_source, mask = img_mask.astype(np.uint8) * 255)
  cat_img = copy_img + cv2.bitwise_and(img_target, img_target, mask=new_mask.astype(np.uint8) * 255)
  # compute regions to be blended
  region_source = (
      max(-offset[0], 0),
      max(-offset[1], 0),
      min(img_target.shape[0] - offset[0], img_source.shape[0]),
      min(img_target.shape[1] - offset[1], img_source.shape[1]))
  region_target = (
      max(offset[0], 0),
      max(offset[1], 0),
      min(img_target.shape[0], img_source.shape[0] + offset[0]),
      min(img_target.shape[1], img_source.shape[1] + offset[1]))
  region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

  # clip and normalize mask image
  img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
  img_mask[img_mask == 0] = False
  img_mask[img_mask != False] = True

  # determines the diagonals on the coefficient matrix
  positions = np.where(img_mask)
  # setting the positions to be in a flatted manner
  positions = (positions[0] * region_size[1]) + positions[1]

  # row and col size of coefficient matrix
  n = np.prod(region_size)

  main_diagonal = np.ones(n)
  main_diagonal[positions] = 4
  diagonals = [main_diagonal]
  diagonals_positions = [0]

  # creating the diagonals of the coefficient matrix
  for diagonal_pos in [-1, 1, -region_size[1], region_size[1]]:
      in_bounds_indices = None
      if np.any(positions + diagonal_pos > n):
          in_bounds_indices = np.where(positions + diagonal_pos < n)[0]
      elif np.any(positions + diagonal_pos < 0):
          in_bounds_indices = np.where(positions + diagonal_pos >= 0)[0]
      in_bounds_positions = positions[in_bounds_indices]

      diagonal = np.zeros(n)
      diagonal[in_bounds_positions + diagonal_pos] = -1
      diagonals.append(diagonal)
      diagonals_positions.append(diagonal_pos)
  A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')

  # create poisson matrix for b
  P = pyamg.gallery.poisson(img_mask.shape)

  # get positions in mask that should be taken from the target
  inverted_img_mask = np.invert(img_mask.astype(np.bool)).flatten()
  positions_from_target = np.where(inverted_img_mask)[0]

  # for each layer (ex. RGB)
  for num_layer in range(img_target.shape[2]):
      # get subimages
      t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
      s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
      t = t.flatten()
      s = s.flatten()

      # create b
      b = P * s
      b[positions_from_target] = t[positions_from_target]

      # solve Ax = b
      x = scipy.sparse.linalg.spsolve(A, b)

      # assign x to target image
      x = np.reshape(x, region_size)
      x = np.clip(x, 0, 255)
      x = np.array(x, img_target.dtype)
      # print(num_layer)
      # imshow(x)
      # imshow(img_source[:, :, num_layer])
      # x = match_histograms(x, img_source[:, :, num_layer], multichannel=True)
      # imshow(x)
      # e()
      img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x
      # img_target[region_target[1]:region_target[3], region_target[0]:region_target[2], num_layer] = x
  return img_target



def extend_boundary(mask, src, inc):
  mask_orig = np.copy(mask)
  mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
  weight = torch.Tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
  ]).type(mask.dtype)
  mask = F.conv2d(F.pad(mask, (1, 1, 1, 1)), weight.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()
  src_2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  src_2 = cv2.normalize(src_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  # imshow(mask * 255)
  n_mask = cv2.resize(mask, (int(mask.shape[0] * 1.1), int(mask.shape[1] * 1.1)), interpolation=cv2.INTER_CUBIC)
  # imshow(n_mask * 255)
  diff_r = n_mask.shape[0] - mask.shape[0]
  diff_c = n_mask.shape[1] - mask.shape[1]
  new_mask = n_mask[diff_r//2:n_mask.shape[0] - diff_r//2, diff_c//2:n_mask.shape[1] - diff_c//2]
  new_mask = cv2.resize(new_mask, (mask.shape[0], mask.shape[1]))
  boundary_mask = np.where(new_mask==4, 0, new_mask)
  new_mask = cv2.normalize(new_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  boundary_mask = np.where(mask==4, 0, mask)
  new_mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  boundary_mask = cv2.normalize(boundary_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  return new_mask, boundary_mask


def get_masks(im):
  cfg = get_cfg()
  cfg.MODEL.DEVICE='cpu'
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)
  segments = np.asarray(outputs["instances"].pred_masks)
  boxes = outputs["instances"].pred_boxes
  return segments


def combine_img(configs):
  source_path = configs["args"]["source"]
  target_path = configs["args"]["target"]
  output_path = configs["args"]["output"]
  img_size = configs["args"]["img_size"]

  i = 0
  for tgt in tqdm(os.listdir(target_path)):
    target = os.path.join(target_path, tgt)
    for src in os.listdir(source_path):
      source = os.path.join(source_path, src)
      im = cv2.imread(source)
      tgt = cv2.imread(target)
      rs, cs, _ = im.shape
      rd, cd, _ = tgt.shape
      im  = cv2.resize(im, (img_size, img_size))
      tgt  = cv2.resize(tgt, (img_size, img_size))
      segments = get_masks(im)
      for mask in (segments.astype(np.uint8)):
        mask, boundary_mask = extend_boundary(mask, im, 30)
        if np.count_nonzero(mask==1) < configs["args"]["threshold"]:
          print(np.count_nonzero(mask==1))
          continue
        if configs["args"]["blend_type"] == "copy_paste":
          new_mask = 1 - mask
          cat_img = cv2.bitwise_and(im, im, mask = mask) + cv2.bitwise_and(tgt, tgt, mask = new_mask)
        elif configs["args"]["blend_type"] == "poisson":
          cv2.imwrite("./input/mask_plane.png", mask * 255)
          # new_mask= 1 - mask
          # cat_img = cv2.bitwise_and(im, im, mask = mask.astype(np.uint8) * 255) + cv2.bitwise_and(tgt, tgt, mask = new_mask.astype(np.uint8) * 255)
          # cat_img = blend(np.copy(cat_img), np.copy(im), np.copy(boundary_mask))
          # cat_img = solve_poisson(np.copy(tgt), np.copy(im), np.copy(mask), boundary_mask)
          cat_img = blend(np.copy(tgt), np.copy(im), np.copy(mask))
          # imshow(cat_img)
          # print(im.shape, tgt.shape, mask.shape)
          # cat_img = cv2.seamlessClone(im, tgt, mask * 255, (rd//2, cd//2), cv2.MIXED_CLONE)
        cv2.imwrite(f"{output_path}/{i}.png", cat_img)
        i+=1
