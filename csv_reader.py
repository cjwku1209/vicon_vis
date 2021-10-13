import csv
import numpy as np
import cv2
from helper import *

# Setup
file = "data/"
frame_number= 500
participant= "pp05"
side = False
idx = [0, 8, 10, 12, 7, 3, 5, 1, 4, 6, 2, 11, 13, 9] # change index numbers of vicon joints

# Initialization
camera_id = "2127314"
if side:
	camera_id = "2133985"

vicon_csv = file+participant+"/"+participant+"_score.csv"
camera_file = file+participant+"/"+participant+".xcp"
cap = cv2.VideoCapture(file+participant+"/"+participant+"."+camera_id+".avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
success, image = cap.read()

with open(vicon_csv, newline='') as csvfile:
	frames = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))
	joint_name = list(filter(lambda x : x != '', frames[2][0].split(",")))
    # joint_name = list(filter(lambda x : x != '', frames[1 + frame_number][0].split(",")))
	# frame= list(map(lambda x : float(x) if x != "" else 0, frames[5 + frame_number][0][4:].split(",")))
	frame= list(map(lambda x : float(x) if x != "" else 0, frames[6 + frame_number][0][6:].split(",")))
	joints = {joint_name[i]: frame[i*3: i*3+3] for i in range(0, len(joint_name))}

print(len(joints))
joints_3d = to_matrix(frame, 3)
joints_3d = np.hstack((joints_3d, np.ones((joints_3d.shape[0], 1))))
joints_3d = np.insert(joints_3d, 0, (joints_3d[6] + joints_3d[7]) / 2, axis=0)
joints_3d = joints_3d[idx]
joints_2d = project2d(joints_3d, camera_file, camera_id)


img = draw_joints(image, joints_2d)
cv2.imshow('Vicon', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
