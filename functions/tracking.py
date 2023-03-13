from signal import SIG_DFL
from sqlite3 import threadsafety
from matplotlib.cbook import sanitize_sequence
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math as ma
from pyrsistent import m


class Interpolation():
    def __init__(self):
        self.object_pos = []

    def prediction(self,object,object_vel,object_acc,time):

        for i in range(len(object)):
            print(object[i][-1])
            print(object_vel[i][-1])
            print(object_acc[i][-1])
            self.object_pos.append(object[i][-1]+object_vel[i][-1]*time + 0.5*time*time*object_acc[i][-1])
        return self.object_pos

    def reset(self):
        self.object_pos = []

    
class Track(Interpolation):
    def __init__(self):
        #Interpolation.__init__(self)
        super().__init__()
        self.objects = []
        self.object_pos = []
        self.object_vel =[]
        self.object_acc = []
        self.object_cluster =[]

    @staticmethod
    def dist(centroid1,centroid2):
        calc = np.square(np.subtract(centroid1,centroid2)).mean() 
        thres =  0.2
        if calc < thres : 
            same = True
        else :
            same = False
        return same

    @staticmethod
    def centroid(Points):
        m = len(Points)
        coodr = [0,0,0]
        for i in range(len(Points)):
            coodr = np.add(coodr,Points[i])
        coodr = coodr/m
        return coodr

    def clusters(self,Cluster):
        centroids = []
        for i in range(len(Cluster)):
            centroids.append(self.centroid(Cluster[i]))
        return centroids,Cluster

    def register_acc(self):
        for i in range(len(self.object_vel)):
            if len(self.object_vel[i]) > 1:
                self.object_acc[i] = np.concatenate((self.object_acc[i],[(self.object_vel[i][-1]-self.object_vel[i][-2])]),axis = 0)
    
    def register_velocity(self):
        # Need to add the condition to fulfil the velocity if the intial velocity is 0
        for i in range(len(self.objects)):
            #print(len(self.object_vel[i]))
            if len(self.objects[i]) > 1:
                self.object_vel[i] = np.concatenate((self.object_vel[i],[(self.objects[i][-1]-self.objects[i][-2])]),axis = 0)
        return self.object_vel
    
    def predict(self,time,method):
        object_pos = []
        if method== 'Interpolation' :
            object_pos = Interpolation().prediction(self.objects,self.object_vel,self.object_acc,time)
            Interpolation().reset()
        return object_pos

    def register_object(self,Clusters,thres):
        centers,Clusters = self.clusters(Clusters)
        flag = False
        done = []
        for i in range(len(centers)):
            flag = False
            for j in range(len(self.objects)):
                #print(np.positive(centers[i] - self.objects[j][-1]))
                if max(np.absolute(centers[i] - self.objects[j][-1])) < thres :
                    print('throw')
                    print(len(self.objects[j]))
                    print(len(self.object_cluster[j]))
                    self.objects[j] = np.concatenate((self.objects[j],[centers[i]]),axis = 0)
                    self.object_cluster[j].append(Clusters[i])
                    # self.object_cluster[j] = np.column_stack((self.object_cluster[j],[Clusters[i]]))
                    # self.object_cluster[j] = np.concatenate((self.object_cluster[j],[Clusters[i]]))
                    print(len(self.objects[j]))
                    print(len(self.object_cluster[j]))
                    print('catch')
                    flag = True
                    # done.append(j)
            if flag==False:
                self.objects.append([centers[i]])
                self.object_cluster.append([Clusters[i]])
                self.object_vel.append([[0,0,0]])
                self.object_acc.append([[0,0,0]])
            self.register_velocity()
            self.register_acc()
        # print(self.objects)
        # for obj in range(len(self.objects)):
        #     if obj not in done:
        #         self.predict(self.dt,Interpolation)
        print('the length of the cluster and the objects are ')
        print(len(self.objects[0]))
        print(len(self.object_cluster[0]))
        return self.objects,self.object_cluster 

    









