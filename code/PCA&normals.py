# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 00:24:52 2020

@author: 123
"""

# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
def make_line_pcd(dv,meanData):
    L = 400
    v0 = dv[:,0]
    v1 = dv[:,1]
    v2 = dv[:,2]
    triangle_points = np.array([[meanData[0],meanData[1],meanData[2]],[v0[0]*L,v0[1]*L,v0[2]*L],[v1[0]*L/2
    ,v1[1]*L/2,v1[2]*L/2],[v2[0]*L/3,v2[1]*L/3,v2[2]*L/3]],dtype = np.float32)
    lines = [[0,1],[0,2],[0,3]]
    colors = [[0,0,1],[1,0,1],[0,0,0]]
     # 定义三角形三条连接线
    line_pcd = o3d.LineSet()
    line_pcd.lines = o3d.Vector2iVector(lines)
    line_pcd.colors = o3d.Vector3dVector(colors)
    line_pcd.points = o3d.Vector3dVector(triangle_points)
    return line_pcd
# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
 
def PCA(data, correlation=False, sort=True):

    meanData = np.mean(data,axis=0)
    X = data - meanData
    N = X.shape[0]
    covX = np.cov(X,rowvar=0)
    eigenvalues,eigenvectors = np.linalg.eig(covX)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]
    return eigenvalues,eigenvectors,meanData

def main():
    global point_cloud_o3d
    global points
    '''
    cat_index = 10
    root_dir = r'E:\3D点云数据集\ModelNet40\car\test\car_0198.off'#数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train',cat[cat_index]+'_0001.ply')
    '''
    filename = r'E:/3DPointsDataset/ModelNet40/car/test/car_0198.off'#数据集路径
    #加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d",mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) #显示原始点云    
    #从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:',points.shape[0])    
    #用PCA分析点云主方向
    _,v,meanData = PCA(points)
    point_cloud_vector = v[:,2]#点云主方向对应的向量
    print("the main orientation of this pointcloud is:",point_cloud_vector)
    line_pcd = make_line_pcd(v,meanData)
    o3d.visualization.draw_geometries([point_cloud_o3d])
    o3d.visualization.draw_geometries([line_pcd]) #在显示窗口按n可看到法向量
        
if __name__ == '__main__':
    main()
       
#循环计算每个点的法向量
pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)  #对点云建立kd树 方便搜索
normals = []
# print(point_cloud_o3d)  #geometry::PointCloud with 10000 points.
print(points.shape[0]) #10000
for i in range(points.shape[0]):
    # search_knn_vector_3d函数 ， 输入值[每一点，x]      返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
    [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 10)  # 10 个临近点
    # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
    k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]  #找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
    w, v, z = PCA(k_nearest_point)
    normals.append(v[:, 2])
point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([point_cloud_o3d])
