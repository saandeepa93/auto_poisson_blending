import matplotlib.pyplot as plt
import cv2
import numpy as np

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

from sys import exit as e

def imshow(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def match_hist(reference, image):
  image = cv2.imread("./output/blend_0.png")
  reference = cv2.imread("./input/jet_3.png")

  image = cv2.resize(image, (512, 512))
  reference = cv2.resize(reference, (512, 512))

  mask = cv2.cvtColor(cv2.imread("input/mask_plane.png"), cv2.COLOR_BGR2GRAY)
  mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  new_mask = 1-mask
  new_mask = cv2.normalize(new_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  image = cv2.bitwise_and(image, image, mask = mask.astype(np.uint8) * 255)
  reference = cv2.bitwise_and(reference, reference, mask = mask.astype(np.uint8) * 255)
  print(np.amax(new_mask), new_mask.dtype, new_mask.shape)




  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

  matched = match_histograms(image, reference, multichannel=True)

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                      sharex=True, sharey=True)
  for aa in (ax1, ax2, ax3):
      aa.set_axis_off()

  ax1.imshow(image)
  ax1.set_title('Source')
  ax2.imshow(reference)
  ax2.set_title('Reference')
  ax3.imshow(matched)
  ax3.set_title('Matched')

  plt.tight_layout()
  plt.show()
  return matched
match_hist(None, None)