# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:27:15 2020

@author: 123
"""

import open3d as o3d
import numpy as np
#读取电脑中的 ply 点云文件
source = o3d.read_point_cloud("E:/3DPointsDataset/bunny/data/top2.ply")  #source 为需要配准的点云
target = o3d.read_point_cloud("E:/3DPointsDataset/bunny/data/top3.ply")  #target 为目标点云

#为两个点云上上不同的颜色
source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#创建一个 o3d.visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()

#将两个点云放入visualizer
vis.add_geometry(source)
vis.add_geometry(target)

#让visualizer渲染点云
vis.update_geometry()
vis.poll_events()
vis.update_renderer()
#展示原始点云
vis.run()

#开始ICP配准
#读取电脑中的 ply 点云文件
threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,1,0],   # 变换矩阵
                         [0,1,0,0],   
                         [0,0,1,0],   
                         [0,0,2,1]])

#运行icp
#TransformationEstimationPointToPoint计算point-to-point ICP的残差和雅可比矩阵的函数
reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())  

#fitness：对这次配准的打分，越大越好； 
#inlier_rmse：root of covariance, 也就是所有匹配点之间的距离的总和除以所有点的数量的平方根，越小越好；
#correspondence_size 代表配准后点云a与点云b点云中吻合的点的数量。
print(reg_p2p)
 
#将我们的矩阵依照输出的变换矩阵进行变换 
source.transform(reg_p2p.transformation)

#创建一个 o3d.visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()

#将两个点云放入visualizer
vis.add_geometry(source)
vis.add_geometry(target)

#让visualizer渲染点云
vis.update_geometry()
vis.poll_events()
vis.update_renderer()
#展示配准后的点云
vis.run()