# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:37:17 2020

@author: 123
"""
start = time.clock()
import open3d as o3d 
import time
import multiprocessing
import os
from multiprocessing import Process

start = time.time()
class MyProcess(multiprocessing.Process):
    def __init__(self, func, args, name=''):
        multiprocessing.Process.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
'''
def spawn_n_processes(n, target):

    threads = []

    for _ in range(n):
        thread = Process(target=target)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def test(target, number=10, spawner=spawn_n_processes):
    """
    分别启动 1, 2, 3, 4 个控制流，重复 number 次，计算运行耗时
    """

    for n in (1, 2, 3, 4, ):

        start_time = time.time()
        for _ in range(number):  # 执行 number 次以减少偶然误差
            spawner(n, target)
        end_time = time.time()

        print('Time elapsed with {} branch(es): {:.6f} sec(s)'.format(n, end_time - start_time))
'''
#from pyntcloud import PyntCloud 
#import pcl
#start = time.clock()
#可视化覆铜板并进行渲染
pcd = o3d.io.read_point_cloud("F:/3Dproject/3Dproject_CCL/data&code/data-dcb-sample-202008 - Cloud.ply")
o3d.visualization.draw_geometries([pcd])
#test(pcd.voxel_down_sample, spawner=spawn_n_processes)
#total_time = time.time() - start
#print(total_time)
#体素大小为0.01时降采样
#start = time.time()
process = []
t = MyProcess(pcd.voxel_down_sample, (0.01,), pcd.voxel_down_sample.__name__)
process.append(t)
for i,j in enumerate(process):
    process[i].start()
for i,j in enumerate(process):
    process[i].join(0)
for i,j in enumerate(process):
    voxel_down_pcd = process[i].get_result() 
#voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
o3d.visualization.draw_geometries([voxel_down_pcd])
total_time = time.time() - start
print(total_time)
print(voxel_down_pcd)
print(process)
print(t)
#from pyntcloud import PyntCloud
#import pcl
#start = time.clock()
#可视化覆铜板并进行渲染
pcd = o3d.io.read_point_cloud("F:/3Dproject/3Dproject_CCL/data&code/data-dcb-sample-202008 - Cloud.ply")
o3d.visualization.draw_geometries([pcd])
#体素大小为0.01时降采样
#voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
test(pcd.voxel_down_sample, voxel_size=0.01, spawner=spawn_n_processes)
o3d.visualization.draw_geometries([voxel_down_pcd])
#Every 50th points are selected
#uni_down_pcd = pcd.uniform_down_sample(every_k_points=50)
#o3d.visualization.draw_geometries([uni_down_pcd])
#total_time = time.clock() - start
#print(total_time)


#start = time.clock()
def display_inlier_outlier(cloud, ind):
    global inlier_cloud
    global points
    global inlier_data
    inlier_cloud = cloud.select_by_index(ind)
#    outlier_cloud = cloud.select_by_index(ind, invert=True) 
#    ptcloud = o3d.io.read_point_cloud(inlier_cloud)
#    Data[:,0]= float(ptcloud.Location[1:5:end,0])   #提取所有点的三维坐标
#    Data[:,1]= float(ptcloud.Location[1:5:end,1])
#    Data[:,2]= float(ptcloud.Location[1:5:end,2]) 
#    point_cloud_pynt = PyntCloud.from_file(inlier_cloud)
    #从点云中获取点，只对点进行处理
#    inlier_data = point_cloud_pynt.points
    #显示外点(红色)和内点(灰色)
#    outlier_cloud.paint_uniform_color([1, 0, 0])
#    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
#    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    o3d.visualization.draw_geometries([inlier_cloud])  #显示去掉噪音后的点云 
    
cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
display_inlier_outlier(voxel_down_pcd, ind)

#cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=150, radius=0.8)
#display_inlier_outlier(voxel_down_pcd, ind)


'''
#import comtypes.client
import os

def convert_files_in_folder(ply):
    deck = o3d.io.read_point_cloud(ply)
    deck.SaveAs(inlier_cloud)
    outputName = inlier_cloud + '.ply'
    deck.Close()
    fullpath = os.path.join('F:\3Dproject', outputName)
    return fullpath
filename = convert_files_in_folder('inlier_cloud')
point_cloud_pynt = PyntCloud.from_file(filename)
#从点云中获取点，只对点进行处理
inlier_data = point_cloud_pynt.points

def write_pcd(points, save_pcd_path):
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z, i, is_g = points[i]
        lines.append('{:.6f} {:.6f} {:.6f} {}'.format( \
            x, y, z, i))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))

def write_pcd(points, save_pcd_path):
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(len(points), len(points)) + '\n')
        np.savetxt(f, points, delimiter = ' ', fmt = '%f %f %f %d')

f=open('inlier_cloud','r')
point=inlier_cloud.read()
'''

import numpy as np
import random 
import math
import sys
#import open3d
#import os
#import struct
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

'''
def read_velodyne_bin(path):
    
    :param path:
    :return: homography matrix of the point cloud, N*3
    
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

path="E:/3DPointsDataset/Tools_RosBag2KITTI_master/pcd2bin/bin/000000.bin"
origindata=read_velodyne_bin(path)
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(origindata)
open3d.visualization.draw_geometries([pcd])
'''

#用最小二乘和ransac拟合平面
def PlaneLeastSquare(X:np.ndarray):
    #z=ax+by+c,return a,b,c
    A=X.copy()
    b=np.expand_dims(X[:,2],axis=1)
    A[:,2]=1
    
    #通过X=(AT*A)-1*AT*b直接求解

    A_T = A.T
    A1 = np.dot(A_T,A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    x= np.dot(A3, b)
    return x


def PlaneRANSAC(X:np.ndarray,tao:float,e=0.4,N_regular=100):
    #return plane ids
    s = X.shape[0]
    count = 0
    p = 0.99
    dic = {}
    
    #确定迭代次数
    if math.log(1-(1-e)**s)<sys.float_info.min:
        N = N_regular
    else:
        N = math.log(1-p)/math.log(1-(1-e)**s)
    #开始迭代
    while count < N:
        ids = random.sample(range(0,s),3)
#        Points = X[ids]
        p1,p2,p3 = X[ids]
        #判断是否共线
        L = p1 - p2
        R = p2 - p3
        if 0 in L or 0 in R:
            continue
        else:
            if L[0]/R[0] == L[1]/R[1] == L[2]/R[2]:
                continue
        
        #计算平面参数
        a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
        b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
        c = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);     
        d = 0 - (a * p1[0] + b*p1[1] + c*p1[2]);
        
        dis=abs(a*X[:,0]+b*X[:,1]+c*X[:,2]+d)/(a**2+b**2+c**2)**0.5
        
        idset=[]
        for i ,d in enumerate(dis):
            if d <tao:
                idset.append(i)
        
        
        #再使用最小二乘法
        p=PlaneLeastSquare(X[idset])
        a,b,c,d=p[0],p[1],-1,p[2]
        
        
        dic[len(idset)]=[a,b,c,d]
        
        if len(idset)>s*(1-e):
            break
        
        count+=1
    
    parm=dic[max(dic.keys())]
    a,b,c,d=parm
    dis=abs(a*X[:,0]+b*X[:,1]+c*X[:,2]+d)/(a**2+b**2+c**2)**0.5
        
    idset=[]
    for i ,d in enumerate(dis):
        if d <tao:
            idset.append(i)
    return np.array(idset)

#total_time = time.clock() - start
#print(total_time)

pro1 = multiprocessing.Process(target=PlaneLeastSquare, args=())
#start = time.clock()
#找出平面点
inlier_data = np.asarray(inlier_cloud.points)
planeids=PlaneRANSAC(inlier_data,0.01)
planedata=inlier_data[planeids]
planepcd = o3d.geometry.PointCloud()
planepcd.points = o3d.utility.Vector3dVector(planedata)

#c=[0,0,255]
#cs=np.tile(c,(planedata.shape[0],1))
#planepcd.colors=o3d.utility.Vector3dVector(cs)

#找出平面外的点
othersids=[]
for i in range(inlier_data.shape[0]):
    if i not in planeids:
        othersids.append(i)
otherdata=inlier_data[othersids]
otherpcd = o3d.geometry.PointCloud()
otherpcd.points = o3d.utility.Vector3dVector(otherdata)
#c=[255,0,0]
#cs=np.tile(c,(otherdata.shape[0],1))
#otherpcd.colors=o3d.utility.Vector3dVector(cs)
o3d.visualization.draw_geometries([planepcd])      #显示平面点
o3d.visualization.draw_geometries([otherpcd])      #显示平面外的点 

total_time = time.clock() - start
print(total_time)

#点到平面距离
class distance:
    def getDistanceBetweenPointAndFace(x,y,z,face):
        #(a,b,c) is the normal vector
        a = (face[1][1]-face[0][1])*(face[2][2]-face[0][2]) - (face[2][1]-face[0][1])*(face[1][2]-face[0][2])
        b = (face[2][0]-face[0][0])*(face[1][2]-face[0][2]) - (face[1][0]-face[0][0])*(face[2][2]-face[0][2])
        c = (face[1][0]-face[0][0])*(face[2][1]-face[0][1]) - (face[2][0]-face[0][0])*(face[1][1]-face[0][1])
     
        t = (a*face[0][0] + b*face[0][1] + c*face[0][2] - (a*x + b*y + c*z))/(a**2+b**2+c**2)
        # t_same1 = (a*face[1][0] + b*face[1][1] + c*face[1][2] - (a*x + b*y + c*z))/(a**2+b**2+c**2)
        # t_same2 = (a*face[2][0] + b*face[2][1] + c*face[2][2] - (a*x + b*y + c*z))/(a**2+b**2+c**2)
        # assert (t-t_same1)<0.0000001 and (t-t_same2)<0.0000001
     
        #X0,Y0,Z0 is the projection point
        X0 = x + a*t
        Y0 = y + b*t
        Z0 = z + c*t
     
        if  distance.getInnerProduct(face[0][0]-X0, face[0][1]-Y0, face[0][2]-Z0, face[1][0]-X0, face[1][1]-Y0, face[1][2]-Z0)<=0\
            and distance.getInnerProduct(face[0][0]-X0, face[0][1]-Y0, face[0][2]-Z0, face[2][0]-X0, face[2][1]-Y0, face[2][2]-Z0)<=0\
            and distance.getInnerProduct(face[1][0]-X0, face[1][1]-Y0, face[1][2]-Z0, face[2][0]-X0, face[2][1]-Y0, face[2][2]-Z0)<=0:
            # print(str(X0)+' '+str(Y0)+' '+ str(Z0)+'On face')
            return distance.getDistanceBetweenTwoPoints(x,y,z,X0,Y0,Z0)
        else:
            # print(str(X0)+' '+str(Y0)+' '+ str(Z0)+'Not on face')
            return min([distance.getDistanceBetweenPointAndLine(face[0][0],face[0][1],face[0][2],face[1][0],face[1][1],face[1][2],x,y,z),
                         distance.getDistanceBetweenPointAndLine(face[0][0],face[0][1],face[0][2],face[2][0],face[2][1],face[2][2],x,y,z),
                         distance.getDistanceBetweenPointAndLine(face[1][0],face[1][1],face[1][2],face[2][0],face[2][1],face[2][2],x,y,z)])
     
    #refer to https://blog.csdn.net/gf771115/article/details/26721055/
    #X3,Y3,Z3 is the point
    def getDistanceBetweenPointAndLine(X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3):
        k = ((X3-X1)*(X2-X1)+(Y3-Y1)*(Y2-Y1)+(Z3-Z1)*(Z2-Z1))/(distance.getDistanceBetweenTwoPoints(X1,Y1,Z1,X2,Y2,Z2)**2)
        k_same = distance.getInnerProduct(X3-X1,Y3-Y1,Z3-Z1, X2-X1,Y2-Y1,Z2-Z1)/(distance.getDistanceBetweenTwoPoints(X1,Y1,Z1,X2,Y2,Z2)**2)
        assert k==k_same
     
        X0 = X1 + k*(X2-X1)
        Y0 = Y1 + k*(Y2-Y1)
        Z0 = Z1 + k*(Z2-Z1)
        #X0,Y0,Z0 is Projection point
     
        # is on line.
        if distance.getInnerProduct(X1-X0, Y1-Y0, Z1-Z0, X2-X0, Y2-Y0, Z2-Z0)<=0:
            return distance.getDistanceBetweenTwoPoints(X3,Y3,Z3,X0,Y0,Z0)
        else:# not on line.
            return min([distance.getDistanceBetweenTwoPoints(X1,Y1,Z1,X3,Y3,Z3),distance.getDistanceBetweenTwoPoints(X2,Y2,Z2,X3,Y3,Z3)])
        
    def getDistanceBetweenTwoPoints(X1,Y1,Z1,X2,Y2,Z2):
        return ((X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2)**0.5
     
    def getInnerProduct(DetaX1,DetaY1,DetaZ1,DetaX2,DetaY2,DetaZ2):
        return DetaX1*DetaX2 + DetaY1*DetaY2 + DetaZ1*DetaZ2
     

m = np.floor(0.82/0.2) + 1 #每一行有m个bounding_box
n = np.floor(0.756/0.2) + 1 #每一列有n个bounding_box
#times = m * n  #共有times个bounding_box    
pcd_tree = o3d.geometry.KDTreeFlann(planepcd)  #对平面建立KDtree搜索
[k, idx, _] = pcd_tree.search_radius_vector_3d(inlier_cloud.points[0], 0.2) #对该点半径为0.2的范围内进行查找
#    labels = [[] for i in range(times)]
bounding_data = o3d.utility.Vector3dVector(planedata[idx])
bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bounding_data)
labels = []   #存储bounding_box的位置
diff = []     #存储平面上的点到bounding_box中心的距离
height = []   #存储平面外的点到平面
row = 0
while row < n:
    row += 1
    col = 1
    while col <= m:
        #为筛出气泡先对区域进行分割（保证有交叉区域）
        #对其中一个区域进行搜索
#        for i in range(planedata.shape[0]):
        box_center = o3d.geometry.AxisAlignedBoundingBox.get_center(bounding_box) #
        [k, idx, _] = pcd_tree.search_radius_vector_3d(box_center, 0.2) #对该点半径为0.2的范围内进行查找
        #    labels = [[] for i in range(times)]
#        labels = []
#        labels1 = []
        #计算距离（半径）
        for j in range(k):
#            diff = []
            diff.append(np.linalg.norm(box_center - inlier_data[j]))
        diff2 = np.max(diff)
        #计算高度
        for l in othersids:
#            height = []
            height.append(distance.getDistanceBetweenPointAndFace(inlier_data[l][0],inlier_data[l][1],inlier_data[l][2],inlier_data))
        max_height = max(height)
        #根据气泡条件确定气泡区域
#        for p,q in enumerate(range(times)):
#            if diff2 >= 0.1 and max_height >= 0.012:         
#                labels.append(p)
        if diff2 >= 0.001 and max_height >= 0.0012:         
            labels.append((row, col))
            print(labels)
            o3d.visualization.draw_geometries([bounding_box,inlier_cloud])
        #挑出气泡区域中的点云
#        for v in inlier_data:
#            labels1.append(v[labels[p]])
#        bubble = bounding_box[labels]
        col += 1
        #对bounding_box平移
        #迭代
#        bounding_box_tx = copy.deepcopy(bounding_box).translate((0.2,0,0))
        bounding_box.translate((0.2,0,0))
    R = inlier_cloud.get_rotation_matrix_from_xyz((np.pi,np.pi,np.pi))
    inlier_cloud.rotate(R,center = (5.158, 4.658, 0.048646))
#    bounding_box_ty = copy.deepcopy(bounding_box).translate((0,0.2,0))
    bounding_box.translate((0,0.2,0))
