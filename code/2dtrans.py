import cv2
import numpy as np
import matplotlib.pyplot as plt
 
image = cv2.imread('E:/3DPointsDataset/KittiSeg-master/data/examples/um_road_000005.png')
rows,cols,channel = image.shape
#对应点
src_points = np.float32([[0,0],[cols-1,0],[0,rows-1]])   #原始图像中的三个点的坐标
dst_points = np.float32([[0,0],[int(0.6*(cols-1)),0],    #变换后的这三个点对应的坐标
                         [int(0.4*(cols-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points,dst_points)  #根据变换前后三个点的对应关系来自动求解仿射变换所需的affine_matrix矩阵（6参数）
affine_image = cv2.warpAffine(image,affine_matrix,(cols,rows)) #三个参数为需要变换的原始图像，仿射变换矩阵affine_matrix以及变换的图像大小 
#cv2.imshow('Original Image',image)
#cv2.imshow('Affine Image',affine_image)
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(affine_image)
plt.show()