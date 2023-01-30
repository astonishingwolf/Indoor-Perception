import numpy as np
from functions.pc_matching import PCMatch
import cv2
import open3d as o3d

def video_cv(frame_list):
    for frame in frame_list:
        cv2.imshow('PC',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def pc2video(pc_list, x_min = 400, x_max = 1000, y_min = 0, y_max = 250 ):
    '''
    the x and y min and max values are to slice a part from the point cloud projection to visualise
    '''
    match_points = PCMatch()
    new_pc = []
    for i in range(len(pc_list)):
        frame = np.nan_to_num(pc_list[i],True)
        frame,_ = match_points.tdproj(frame)
        frame = frame[x_min:x_max,y_min:y_max]
        new_pc.append(frame)

    video_cv(new_pc)

def rec_video(frame_list,filepath):
    '''
    filepath argument must be filled with the path location where the video is to be stored
    '''
    out = cv2.VideoWriter(filepath,cv2.VideoWriter_fourcc('M','J','P','G'), 15, frame_list[0].shape[::-1])
    for i in range(len(frame_list)):
        out.write(frame_list[i])
    out.release()

def pc_viz(pts):
    pcd_list = []
    if type(pts) == list:
        np.random.seed(14)
        colors = np.random.rand(3,len(pts))
        for i in range(len(pts)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[i])
            pcd.paint_uniform_color(colors[:,i])
            pcd_list.append(pcd)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)          #inlier_cloud,
                                    # zoom=0.8,
                                    # front=[-0.4999, -0.1659, -0.8499],
                                    # lookat=[2.1813, 2.0619, 2.0999],
                                    # up=[0.1204, -0.9852, 0.1215])
    
    

