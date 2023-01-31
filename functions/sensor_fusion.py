import open3d as o3d
import numpy as np
import importlib
import matplotlib.pyplot as plt
import h5py
import functions.pc_matching
import functions.mat_to_py
import functions.viz
import functions.pc_registration
import functions.sensor_fusion
import functions.tracking
importlib.reload(functions.tracking)
importlib.reload(functions.sensor_fusion)
importlib.reload(functions.pc_matching)
importlib.reload(functions.mat_to_py)
importlib.reload(functions.viz)
importlib.reload(functions.pc_registration)
from functions.tracking import Track
from functions.pc_matching import PCMatch
from functions.mat_to_py import PCNIData
from functions.viz import pc_viz, video_cv, pc2video
from functions.pc_registration import PointReg
import cv2 as cv
import argparse
import numpy as np
import sklearn as skl
from sklearn.cluster import KMeans
import functions.imageprediction
importlib.reload(functions.imageprediction)
from functions.imageprediction import imagepred
import math as m

class sensorfusion():
    def __init__(self):
        self.i = 0
        self.img_pred = imagepred()
        self.CCenter = Track() 
    
    def projection(self,cluster,image,step,num):
        cluster_homo = []
        for i in cluster[step]:
            cluster_homo.append(np.append(i,1))
        cam_lid_trans = np.load('transformation\\CC_proj_L(1).npy')
        cluster_trans =  np.dot(cam_lid_trans,np.transpose(cluster_homo))
        cluster_trans =  np.transpose(cluster_trans)
        for i in cluster_trans:
            i[0] = i[0]/i[2]
            i[1] = i[1]/i[2]
            i[0] = i[0]/3
            i[1] = i[1]/3
        return i
    
    def solve(bl, tr, p) :
        if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
            return True
        else :
                return False

    def matchID(self,cluster,image,step,num):
        bbox = self.img_pred(image,num,num+1,3)
        Center = self.CCenter.centroid(cluster[step])
        cam_lid_trans = np.load('transformation\\CC_proj_L(1).npy')
        Center = np.append(Center,1)
        cluster_trans =  np.dot(cam_lid_trans,np.transpose(Center))
        cluster_trans =  np.transpose(cluster_trans)
        [x,y,w,h] = bbox
        i = cluster_trans
        i[0] = i[0]/i[2]
        i[1] = i[1]/i[2]
        i[0] = i[0]/3
        i[1] = i[1]/3
        x1 = m.trunc(i[1])
        y1 = m.trunc(i[0])
        if self.solve((x,y),(x+w,y+h),(y1,x1)):
            print('This is our required Cluster with center')

    def clear_cluster(self,cluster,image,step,num):
        img = image[num]
        [x,y,w,h] = self.img_pred(image,num,num+1,3)
        cam_lid_trans = np.load('transformation\\CC_proj_L(1).npy')
        for i in cluster[step]:
            i = np.append(i,1)
            cluster_trans =  np.dot(cam_lid_trans,np.transpose(i))
            cluster_trans =  np.transpose(cluster_trans)
            i = cluster_trans
            i[0] = i[0]/i[2]
            i[1] = i[1]/i[2]
            i[0] = i[0]/3
            i[1] = i[1]/3
            x1 = m.trunc(i[1])
            y1 = m.trunc(i[0])
            # print('working')
            print((x,y,x+w,y+h,x1,y1))
            if self.solve((x,y),(x+w,y+h),(y1,x1)):
                print('working')
                img[x1][y1] = [120,5,5]
                img[x1-1][y1+1] = [120,5,5]
                img[x1-1][y1-1] = [120,5,5]
                img[x1+1][y1+1] = [120,5,5]
                img[x1+1][y1-1] = [120,5,5]
        cv.imshow('PC',img)
        cv.waitKey(-1)


    def __call__(self):
        return self