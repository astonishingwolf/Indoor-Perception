import numpy as np

from pc_matching import PCMatch
from mat_to_py import PCNIData
from viz import pc_viz
from pc_registration import PointReg

def hosp_perc():
    # Initialise all the classes
    match_points = PCMatch()
    data_points = PCNIData('C:\\Users\\vndevara\\Desktop\\Naren\\Sensordata\\Jul14-2022_CAM-LCR_LIDAR_applanix',eg=3)
    reference_frame = np.load('reference_pc.npy')

    # Get the data (Online/Offline)
    data_list = data_points('pc')
    objects_register = []
    register_bool = []

    clusters_list = match_points(reference_frame,frame[0])
    objects_register += clusters_list
    register_bool += [False for i in range(len(clusters_list))]
    objects_register += clusters_list
    for frame in data_list:
        # Get Clusters
        clusters_list = match_points(reference_frame,frame)
        
        


