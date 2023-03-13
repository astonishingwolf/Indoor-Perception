import numpy as np
import open3d as o3d
import copy
from functions.pc_matching import PCMatch
import math


class PointReg():
    def __init__(self,T_init = np.eye(4),icp_threshold = 0.125,match_threshold=1e-3,voxel_size = 0.007,GICP = True , Ransac = False) -> None:
        # for point to point ICP use icp threshold as 0.02
        #for gernealised Icp use icp threshold as 0.12
        self.source = None
        self.target = None
        self.T = T_init
        self.icp_threshold = icp_threshold
        self.match_threshold = match_threshold
        self.voxel_size = voxel_size
        self.GIcp = GICP
        self.RANSAC = Ransac
        # self.pc_agg = None
        # self.corr_set = None

    def reset_trans(self,s_pc,t_pc):
        self.T = np.eye(4)
        self.T[:3,3] = np.mean(t_pc,0)-np.mean(s_pc,0)

    def pc_reset(self,s_pc,t_pc):
        self.source = o3d.geometry.PointCloud()
        self.source.points = o3d.utility.Vector3dVector(s_pc)
        self.source.estimate_normals()
        self.target = o3d.geometry.PointCloud()
        self.target.points = o3d.utility.Vector3dVector(t_pc)
        self.target.estimate_normals()
        print('saved transformation')
        # self.T = np.eye(4)
        # print(self.T)
        # self.T[:3,3] = np.mean(t_pc,0)-np.mean(s_pc,0)
        print(self.T)

    @staticmethod
    def init_transform(s_pc,t_pc):
        s_max_ind  = np.argmax(s_pc[:,1])
        s_min_ind  = np.argmin(s_pc[:,1])
        t_max_ind  = np.argmax(t_pc[:,1])
        t_min_ind  = np.argmin(t_pc[:,1])
        vec1 = s_pc[s_min_ind]-s_pc[s_max_ind]
        vec2 = t_pc[t_min_ind]-t_pc[t_max_ind]
        angle = math.acos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        return angle

    def preprocess_point_cloud(self,pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        #pcd_down = self.source.voxel_down_sample(self.voxel_size)
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd, pcd_fpfh

    def execute_global_registration(self,source_down, target_down, source_fpfh,
                                target_fpfh):
        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            #o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(False),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.99999))
        print(result)
        return result

    def execute_fast_global_registration(self,source_down, target_down, source_fpfh,
                                     target_fpfh):
        distance_threshold = self.voxel_size * 0.5
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result


    def draw_registration_result(self):
        source_temp = copy.deepcopy(self.source)
        self.source.paint_uniform_color([1, 0.706, 0])
        self.target.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(self.T)
        o3d.visualization.draw_geometries([source_temp, self.target])

    def pc_reg(self,result):
        #self.T = trans
        if self.RANSAC == True:
            print('Applying RANSAC based Transformation')
            transform =  result.transformation
        else:
            transform = self.T
        if self.GIcp == True : 
            print("Apply Generalised ICP")
            reg_p2l = o3d.pipelines.registration.registration_generalized_icp(
            self.source, self.target, self.icp_threshold, 
            transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
        else :
            print("Apply Point to Point ICP")
            reg_p2l = o3d.pipelines.registration.registration_icp(
            self.source, self.target, self.icp_threshold, 
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2l)
        print("Transformation after ICP is:")
        print(reg_p2l.transformation)
        self.T = reg_p2l.transformation.copy()
        self.draw_registration_result()
        corr_set = np.asarray(reg_p2l.correspondence_set)
        return corr_set

    def pc_trans(self):
        self.source = self.source.transform(self.T)

    def missing_pts(self,corr_set):
        source_ind = corr_set[:,0]
        target_ind = corr_set[:,1]
        self.pc_trans()
        source_pts = np.asarray(self.source.points)[source_ind]
        target_pts = np.asarray(self.target.points)[target_ind]
        dist = np.linalg.norm(source_pts-target_pts,axis=-1)
        match_false = np.where(dist>self.match_threshold)
        false_ind = corr_set[match_false,:]
        return false_ind.squeeze()

    def pc_update(self,false_ind):
        t_pc = np.asarray(self.target.points)
        s_pc = np.asarray(self.source.points)
        new_points = np.vstack([t_pc,s_pc[false_ind[:,0],:]])
        return new_points
    
    def localisation(self,source_pts,target_pts):
        print('world smalllest violin')
        self.pc_reset(source_pts,target_pts)
        print(self.T)
        # self.T[:3,3] = np.mean(target_pts,0)-np.mean(source_pts,0)
        source_down,source_fpfh = self.preprocess_point_cloud(source_pts)
        target_down,target_fpfh = self.preprocess_point_cloud(target_pts)
        results = self.execute_global_registration(source_down,target_down,source_fpfh,target_fpfh)
        #results = self.execute_fast_global_registration(source_down,target_down,source_fpfh,target_fpfh)
        #print(results)
        corr_set = self.pc_reg(results)
        pc2add = self.missing_pts(corr_set)
        print(pc2add.shape[0],' points added')
        new_pts = self.pc_update(pc2add)
        
        return self.T,new_pts

    def __call__(self, source_pts, target_pts,RANSAC = False):
        # print('wow')
        self.RANSAC = RANSAC
        print('RANSAC BASED METHOD id :',self.RANSAC)
        self.pc_reset(source_pts,target_pts)
        # self.T[:3,3] = np.mean(target_pts,0)-np.mean(source_pts,0)
        source_down,source_fpfh = self.preprocess_point_cloud(source_pts)
        target_down,target_fpfh = self.preprocess_point_cloud(target_pts)
        results = self.execute_global_registration(source_down,target_down,source_fpfh,target_fpfh)
        #results = self.execute_fast_global_registration(source_down,target_down,source_fpfh,target_fpfh)
        #print(results)
        corr_set = self.pc_reg(results)
        pc2add = self.missing_pts(corr_set)
        print(pc2add.shape[0],' points added')
        new_pts = self.pc_update(pc2add)
        
        return new_pts
