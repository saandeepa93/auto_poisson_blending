import matplotlib.pyplot as plt
import cv2

def imshow(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def show(img):
  plt.imshow(img)
  plt.show()
