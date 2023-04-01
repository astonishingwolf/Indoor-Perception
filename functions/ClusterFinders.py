from functions.pc_matching import PCMatch
from functions.tracking import Track
from functions.pc_registration import PointReg
import functions.pc_registration
import importlib
importlib.reload(functions.pc_registration)
from functions.pc_registration import PointReg
class ClusterFinder():

    def __init__(self):
        self.clusters = []
        self.match_points = PCMatch()
        self.tracking = Track()
        self.register = PointReg()

    def registration(self,datalists,object):
        
        reg_pts = datalists[object][0]
        self.register.reset_trans(datalists[object][1],reg_pts)
        print('Initial number of points in cluster: ',reg_pts.shape[0])
        datalists[object] = datalists[object][1:]
        for count,i in enumerate(datalists[object]): 
            # transform = centers[object][count] - centers[object][count-1]   
            # in_t = [[1,0,0,transform[0]],[0,1,0,transform[1]],[0,0,1,transform[2]],[0,0,0,1]] 
            print('types')
            print(type(i))
            print(type(reg_pts))
            # if count > 3:
            #     reg_pts = self.register(i,reg_pts,RANSAC=True)    
            # else:
            #     reg_pts = self.register(i,reg_pts)
            reg_pts = self.register(i,reg_pts)
        print('Total points in aggregated point cloud ',reg_pts.shape[0])
        return reg_pts

    def __call__(self,static_pc,data_list,number_st,number_ed,interval):
        for i in range(number_st,number_ed):
            if i % interval == 0 : 
                self.clusters.append(self.match_points(static_pc,data_list[i]))
        object = []
        cluster = []
        for i in range(len(self.clusters)):
            # print(self.clusters[i])
            objects,cluster = self.tracking.register_object(self.clusters[i],0.68)
        # print(objects)
        # print(cluster)
        return objects,cluster
