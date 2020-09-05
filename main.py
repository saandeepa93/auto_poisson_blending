import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import click
from sys import exit as e

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from modules.util import imshow, show
setup_logger()



@click.command()
@click.option('--source', help="source image path")
@click.option('--target', help="target image path")
@click.option('--output', help="output image name")
def blend(source, target, output):
  im = cv2.imread(source)
  tgt = cv2.imread(target)
  rs, cs, _ = im.shape
  rd, cd, _ = tgt.shape
  im  = cv2.resize(im, (cd, rd))
  # if rs >= rd or cs >= cd:
    # rs = int(rd * 0.75)//2
    # cs = int(cd * 0.75)//2
    # im  = cv2.resize(im, (cs, rs))

  print("source image: ", im.shape)
  print("target image: ", tgt.shape)
  cfg = get_cfg()
  cfg.MODEL.DEVICE='cpu'
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)
  segments = np.asarray(outputs["instances"].pred_masks)
  i = 0
  for mask in (segments.astype(np.uint8) * 255):
    # mask = cv2.bitwise_and(im, im, mask = segment.astype(np.uint8))
    # new_mask= 1 - mask.astype(np.uint8)
    # cat_img = cv2.bitwise_and(im, im, mask = mask.astype(np.uint8)) + cv2.bitwise_and(tgt, tgt, mask = new_mask)
    cat_img = cv2.seamlessClone(im, tgt, mask, (cd//2, rd//2), cv2.NORMAL_CLONE)
    cv2.imwrite(f"./output/blend_{i}.png", cat_img)
    i+=1

  # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  # # imshow(out.get_image()[:, :, ::-1])
  # cv2.imwrite(f"./output/jet.png", out.get_image()[:, :, ::-1])


@click.group()
def main():
  pass


if __name__ == '__main__':
  main.add_command(blend)
  main()
