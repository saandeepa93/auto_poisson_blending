import torch
import numpy as np
from skimage import io
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from sys import exit as e

from modules.util import show, imshow

def get_coeff(s):
  a = np.zeros((s, s))
  np.fill_diagonal(a, 2)
  for i in range(s):
    for j in range(s):
      if j != 0 and j != s:
        a[j, j-1] = -1
    if i != 0 and i != s:
      a[i-1, i] = -1
  return a


def get_rhs(tgt_grad, tgt, src):
  a = np.zeros(src.shape[0])
  j = 0
  for i in range(4, 8):
    if i == 4:
      a[j] = tgt_grad[i] - tgt_grad[i+1] + tgt[i-1]
      # print(f"f{i}: {tgt_grad[i]} - {tgt_grad[i+1]} + {tgt[i-1]} = {tgt_grad[i] - tgt_grad[i+1] + tgt[i-1]}")
    elif i == 7:
      a[j] = tgt_grad[i] - tgt_grad[i+1] + tgt[i+1]
      # print(f"f{i}: {tgt_grad[i]} - {tgt_grad[i+1]} + {tgt[i+1]} = {tgt_grad[i] - tgt_grad[i+1] + tgt[i+1]}")
    else:
      a[j] = tgt_grad[i] - tgt_grad[i+1]
      # print(f"f{i}: {tgt_grad[i]} - {tgt_grad[i+1]} = {tgt_grad[i] - tgt_grad[i+1]}")

    # print("-"*25)
    # print(f"f{i}: {a[j]}")
    j+=1
  return a

if __name__ == '__main__':
  dx = nn.Conv2d(1, 1, (1, 3), 1, bias=False)
  dx.weight = nn.Parameter(torch.Tensor([[[[-1, 1]]]]), False)

  dy = nn.Conv2d(1, 1, (3, 1), 1, bias = False)
  dy.weight = nn.Parameter(torch.Tensor([[[[-1.0], [0.0], [1.0]]]]), False)

  tgt = torch.LongTensor(1, 1, 1, 10).random_(1, 10).type(torch.float)
  print("target: ", tgt)
  src = torch.LongTensor(1, 1, 1, 4).random_(50, 55).type(torch.float)

  tgt_grad = dx(tgt)
  src_grad = dx(src)

  tgt_grad = F.pad(tgt_grad, (1, 0), 'constant', 0)
  src_grad = F.pad(src_grad, (1, 0), 'constant', 0)
  tgt[:, :, :, 4:8] = src
  print("target after gradient: ", tgt)

  mask = get_coeff(src.size(-1))
  rhs = get_rhs(tgt_grad.squeeze().detach().numpy(), tgt.squeeze().detach().numpy(), src.squeeze().detach().numpy())
  tgt[:, :, :, 4:8] = torch.Tensor(np.linalg.solve(mask, rhs)).type(torch.float)
  print("target recalculated: ", tgt)
  e()


  img = io.imread("./input/neutral.png")[:, :, :3]
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0), (1))
  ])
  img = transform(img).unsqueeze(0)

  img_dx = dx(F.pad(img, (1, 1, 0, 0), mode='constant', value=0))
  img_dy = dy(F.pad(img, (0, 0, 1, 1), mode='constant', value=0))
  print("dx: ", img_dx.size())
  print("dy: ", img_dy.size())

  # show(img_dx.squeeze().detach())
  # show(img_dy.squeeze().detach())
  imshow(img_dx.squeeze().detach().numpy())
  imshow(img_dy.squeeze().detach().numpy())
