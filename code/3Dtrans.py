import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d 
from pyntcloud import PyntCloud
from PIL import Image


filename = r'E:/3DPointsDataset/ModelNet40/car/test/car_0198.off'#数据集路径
 #加载原始点云
point_cloud_pynt = PyntCloud.from_file(filename)
point_cloud_o3d = point_cloud_pynt.to_instance("open3d",mesh=False)
#从点云中获取点，只对点进行处理
points = point_cloud_pynt.points
points = np.expand_dims(points, axis=0)
#o3d.visualization.draw_geometries([point_cloud_o3d]) #显示原始点云
_,rows,cols = points.shape
#对应点
#src_points = np.float32([[0,0],[cols-1,0],[0,rows-1]])   #原始图像中的三个点的坐标
#dst_points = np.float32([[0,0],[int(0.6*(cols-1)),0],    #变换后的这三个点对应的坐标
#                         [int(0.4*(cols-1)),rows-1]])
src_points = np.float32([[0,1],[5,45],[6,56]])   #原始图像中的三个点的坐标
dst_points = np.float32([[1,13],[7,0],    #变换后的这三个点对应的坐标
                         [9,21]])
affine_matrix = cv2.getAffineTransform(src_points,dst_points)  #根据变换前后三个点的对应关系来自动求解仿射变换所需的affine_matrix矩阵（6参数）
affine_image = cv2.warpAffine(points,affine_matrix,(cols,rows)) #三个参数为需要变换的原始图像，仿射变换矩阵affine_matrix以及变换的图像大小 
#im = Image.fromarray(np.uint8(affine_image))
#cv2.imshow('Original Image',image)
#cv2.imshow('Affine Image',affine_image)
#print(im.shape)
im = o3d.utility.Vector3dVector(affine_image)
o3d.visualization.draw_geometries([point_cloud_o3d])
o3d.visualization.draw_geometries([im])

plt.subplot(121)
plt.imshow(points)
plt.subplot(122)
plt.imshow(affine_image)
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d 
from pyntcloud import PyntCloud


filename = r'E:/3DPointsDataset/ModelNet40/car/test/car_0198.off'#数据集路径
 #加载原始点云
point_cloud_pynt = PyntCloud.from_file(filename)
point_cloud_o3d = point_cloud_pynt.to_instance("open3d",mesh=False)
#从点云中获取点，只对点进行处理
points = point_cloud_pynt.points
points = np.expand_dims(points, axis=0)
#o3d.visualization.draw_geometries([point_cloud_o3d]) #显示原始点云
_,rows,cols = points.shape
#对应点
src_points = np.float32([[0,1],[5,45],[6,56]])   #原始图像中的三个点的坐标
dst_points = np.float32([[1,13],[7,0],    #变换后的这三个点对应的坐标
                         [9,21]])
affine_matrix = cv2.getAffineTransform(src_points,dst_points)  #根据变换前后三个点的对应关系来自动求解仿射变换所需的affine_matrix矩阵（6参数）
affine_image = cv2.warpAffine(points,affine_matrix,(cols,rows)) #三个参数为需要变换的原始图像，仿射变换矩阵affine_matrix以及变换的图像大小 
#cv2.imshow('Original Image',image)
#cv2.imshow('Affine Image',affine_image)
o3d.visualization.draw_geometries([point_cloud_o3d])
o3d.visualization.draw_geometries([affine_image])


