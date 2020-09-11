# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:44:42 2020

@author: 123
"""

import open3d as o3d 
import numpy as np
from pyntcloud import PyntCloud 

filename = r'E:/3DPointsDataset/ModelNet40/car/test/car_0198.off'#数据集路径
#加载原始点云
point_cloud_pynt = PyntCloud.from_file(filename)
pcd = point_cloud_pynt.to_instance("open3d",mesh=False)
pcd.paint_uniform_color([0.5, 0.5, 0.5])  #把颜色设置成灰色

pcd_tree = o3d.geometry.KDTreeFlann(pcd)  #对点云建立kd树,方便搜索     
pcd.colors[1500] = [1, 0, 0]              #对该点标记为红色

#KNN Method
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)  #对该点查找200个最近点
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]   #对该点200个临近点标记为蓝色

#Radius-NN Method
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2) #对该点半径为0.2的范围内进行查找
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]   #对查找出的点标记为绿色

#显示最终点云
o3d.visualization.draw_geometries([pcd])