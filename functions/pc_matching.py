import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import random
import math as m


class PCMatch():
    def __init__(self, ep = 0.4, min_pts = 30, match_mu = 0.1, diff_nu = 0.1) -> None:
        self.ep = ep
        self.min_pts = min_pts
        self.mu = match_mu
        self.nu = diff_nu

    @staticmethod
    def compute_dist_mat(pc1, pc2):
        N, _ = pc1.shape
        M, _ = pc2.shape
        dist = -2 * np.matmul(pc1, pc2.T)
        dist += np.reshape(np.sum(pc1 ** 2, -1),(N, 1))
        dist += np.sum(pc2 ** 2, -1).reshape(1, M)
        return dist

    @staticmethod
    def tdproj(xyz_p,p=1200,q=600):
        xyz_p = np.nan_to_num(xyz_p,True)
        x = xyz_p[:,0]
        y = xyz_p[:,1]
        z = xyz_p[:,2]
        image = np.ones((p,q))
        point_indices = np.zeros((p,q))
        x_res = (x.max()-x.min())/p
        y_res = (y.max()-y.min())/q

        x_img = np.floor((x.max()-x)/x_res).astype(int)
        y_img = np.floor((y-y.min())/y_res).astype(int)
        # print(y.min(),'min y')

        x_img = np.where(x_img<p,x_img,x_img-1)
        y_img = np.where(y_img<q,y_img,y_img-1)

        z_norm = (z-z.min())/(z.max()-z.min())
        indices = np.arange(z_norm.shape[0])
        order = np.argsort(z_norm)[::-1]
        z_norm = z_norm[order]
        indices = indices[order]
        x_img = x_img[order]
        y_img = y_img[order]
        image[x_img,y_img] = z_norm

        point_indices[x_img,y_img] = indices
        
        return image, point_indices

    @staticmethod
    def dbscan(pts,ep,min_pts,viz=False):
        if type(pts).__module__ == np.__name__:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
        else:
            pcd = pts
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=ep, min_points=min_pts, print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        if viz:
            o3d.visualization.draw_geometries([pcd],
                                            zoom=0.455,
                                            front=[-0.4999, -0.1659, -0.8499],
                                            lookat=[2.1813, 2.0619, 2.0999],
                                            up=[0.1204, -0.9852, 0.1215])

        clusters = []
        for i in range(max_label+1):
            ind = np.where(labels == i)
            clusters.append(pts[ind[0]])

        return clusters

    def pc_match(self,X, Y, mu):
        dist = self.compute_dist_mat(X,Y)
        min_dist_arg = np.argmin(dist,1)
        min_dist = dist[range(dist.shape[0]),min_dist_arg]
        min_bool = min_dist<mu
        return min_dist, min_bool


    def distance(self,point):
        return m.sqrt(point[0]*point[0]+point[1]*point[1]+point[2]*point[2])

    def __call__(self, pc1, pc2):
        
        xy1, xy1_ind = self.tdproj(pc1)
        xy2, xy2_ind = self.tdproj(pc2)
        cond = (xy1-xy2)>self.nu

        red_ind2 = xy2_ind[np.nonzero(xy2*cond)]
        pc2 = pc2[red_ind2.astype(np.int32),:]
        red_ind2 = red_ind2[red_ind2<pc1.shape[0]]
        pc1 = pc1[red_ind2.astype(np.int32),:]
        min_dist, bool_arr = self.pc_match(pc1,pc2,self.mu)
        ind_inc = np.where(bool_arr == False)
        pc2 = pc2[ind_inc]

        # plot the projections
        # fig,ax = plt.subplots(2,2,figsize=(15,15))
        # ax[0,0].imshow(xy1)
        # ax[0,1].imshow(xy2)
        # ax[1,0].imshow(cond)

        pc2_1 = []
        pc2_2 = []
        pc2_3 = []
        final_clusters = []
        for i in range(pc2.shape[0]):
            if self.distance(pc2[i]) < 2:
                pc2_1.append(pc2[i])
            elif self.distance(pc2[i]) < 10:
                pc2_2.append(pc2[i])
            else:
                pc2_3.append(pc2[i])
        # print(len(pc2_1))
        # print(len(pc2_2))
        # print(len(pc2_3))
        final_cluster = self.dbscan(pc2,self.ep,self.min_pts,viz=True)
        # if len(pc2_1):
        #     print('first cluster')
        #     pc2_1 = np.stack(pc2_1,axis=0)
        #     final_clusters += self.dbscan(pc2_1,0.2,20)
        # if len(pc2_2):
        #     print('second cluster')
        #     pc2_2 = np.stack(pc2_2,axis=0)
        #     final_clusters += self.dbscan(pc2_2,0.5,20)
        # if len(pc2_3):
        #     pc2_3 = np.stack(pc2_3,axis=0)
        #     print('third cluster')
        #     final_clusters += self.dbscan(pc2_3,self.ep,10)
        return final_cluster





    