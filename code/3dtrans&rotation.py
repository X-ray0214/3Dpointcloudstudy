#显示原始的点云

#Method1
import open3d as o3d
from pyntcloud import PyntCloud 
point_cloud_pynt = PyntCloud.from_file('E:/3DPointsDataset/ModelNet40/airplane/train/airplane_0001.off')
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
o3d.visualization.draw_geometries([point_cloud_o3d]) 

#Method2
import open3d as o3d 
import numpy as np
import copy
pc = o3d.io.read_point_cloud('E:/3DPointsDataset/bunny/data/top2.ply')
print(pc)
print(np.asarray(pc.points))
o3d.visualization.draw_geometries([pc]) 

#以Method2中的点云为例进行旋转和平移
#translation
pc_tx = copy.deepcopy(pc).translate((1000,0,0))
pc_ty = copy.deepcopy(pc).translate((0,2000,0))
pc_tz = copy.deepcopy(pc).translate((-1000,2000,500))
print(f'Center of pc: {pc.get_center()}')
print(f'Center of pc tx: {pc_tx.get_center()}')
print(f'Center of pc ty: {pc_ty.get_center()}')
print(f'Center of pc tz: {pc_tz.get_center()}')
o3d.visualization.draw_geometries([pc, pc_tx, pc_ty, pc_tz])

#rotation
pc_r = copy.deepcopy(pc)
R = pc.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4))
pc_r.rotate(R, center=(0,0,0))
o3d.visualization.draw_geometries([pc, pc_r])
