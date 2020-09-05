from skimage import io as io2
from torchvision import transforms, io
from torch.nn import functional as F
import torch
from sys import exit as e

import modules.util as util

if __name__ == '__main__':
  transformation = transforms.Compose(
    [
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0), (1))
    ]
  )
  img1 = io2.imread('./input/neutral.png')[:, :, :3]
  # img1 = transformation(img1).unsqueeze(0)

  vid1 = io.read_video("./input/02.mp4", pts_unit='sec')[0][:30:15]
  vid1 = vid1.permute(0, 3, 1, 2)
  vid1_lst = []

  for frame in vid1:
    vid1_lst.append(transformation(frame))
  vid1_ten = torch.stack(vid1_lst)

  print("video size: ", vid1_ten.size())

  dx = torch.tensor([
      [-1, 1],
      [-1, 1]
    ]).type(torch.float)

  dy = torch.tensor([
    [-1, -1],
    [1, 1]
  ]).type(torch.float)


  fxy = torch.tensor([
    [0, -1/4, 0],
    [-1/4, 1, -1/4],
    [0, -1/4, 0]
  ]).type(torch.float)

  fx = F.conv2d(vid1_ten, dx.unsqueeze(0).unsqueeze(0), padding=1)
  fy = F.conv2d(vid1_ten, dy.unsqueeze(0).unsqueeze(0), padding=1)



  fx = torch.sum(fx, dim = 0).unsqueeze(0)
  fy = torch.sum(fy, dim = 0).unsqueeze(0)
  ft = (vid1_ten[1] - vid1_ten[0]).unsqueeze(0)

  for i in range(fx.size(0)):
    util.imshow(fx[i].squeeze().detach().numpy())
    util.imshow(fy[i].squeeze().detach().numpy())
    util.imshow(ft[i].squeeze().detach().numpy())


  e()
