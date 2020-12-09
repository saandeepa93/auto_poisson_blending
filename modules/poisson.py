import numpy as np
import cv2
import torch
from torch import Tensor, nn
from torchvision import transforms
from sys import exit as e

from modules.util import imshow, show


def get_gradient(img):
  pre_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
    ]
  )

  img = pre_transform(img).unsqueeze(0)
  filter = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
  ])
  conv2d = nn.Conv2d(1, 1, bias = False, kernel_size=3, padding=1, stride=1)
  conv2d.weight = nn.Parameter(torch.from_numpy(filter).float().unsqueeze(0).unsqueeze(0))

  for params in conv2d.parameters():
    params.requires_grad = False

  blue_channel = img[:, 0].unsqueeze(1)
  green_channel = img[:, 1].unsqueeze(1)
  red_channel = img[:, 2].unsqueeze(1)

  blue_img = conv2d(blue_channel)
  green_img = conv2d(green_channel)
  red_img = conv2d(red_channel)
  # print("blue: ", torch.min(blue_img), torch.max(blue_img), torch.min(blue_channel), torch.max(blue_channel))
  # print("green: ", torch.min(green_img), torch.max(green_img))
  # print("red: ", torch.min(red_img), torch.max(red_img))
  return [blue_img.squeeze().detach().numpy(), green_img.squeeze().detach().numpy(), red_img.squeeze().detach().numpy()]



def solve_poisson(tgt, src, mask, boundary_mask):
  A_grad = get_gradient(tgt)
  B_grad = get_gradient(src)
  A_norm = cv2.normalize(tgt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  B_norm = cv2.normalize(src, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  print(A_norm.dtype, np.amin(A_norm), np.amax(A_norm))
  print(B_norm.dtype, np.amin(B_norm), np.amax(B_norm))
  print(mask.dtype, np.amin(mask), np.amax(mask))

  for i in range(tgt.shape[2]):
    inv_mask = 1 - mask
    A = A_norm[:, :, i]
    B = B_grad[i]
    H = A * inv_mask + B * mask
    boundary_ind = np.argwhere(boundary_mask!=0)
    imshow(H)
    H[boundary_ind[:, 0], boundary_ind[:, 1]] = A[boundary_ind[:, 0], boundary_ind[:, 1]]
    imshow(H)
    e()


