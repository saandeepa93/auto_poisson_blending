import torch
from torch import nn


class Gradient(nn.Module):
  def __init__(self):
    super(Gradient, self).__init__()
    self.weight = torch.tensor([
      [-1, 1],
      [-1, 1]
    ])
    self.conv2d