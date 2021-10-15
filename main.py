import csv
import numpy as np
import cv2
from helper import *
from process_vibe import *
import numpy as np
import xml.etree.ElementTree as ET
import transforms3d
import csv
import matplotlib.pyplot as plt
import joblib
import torch

# Setup
file = "data/"
frame_number= 500
participant= "pp10"
side = False
fig, ax = plt_init()

# Initialization
camera_id = "2127314"
if side:
	camera_id = "2133985"

vicon_csv = file+participant+"/"+participant+"_score.csv"
# camera_file = file+participant+"/"+participant+".xcp"
camera_file = file+"pp08/pp08.xcp"
vibe_output_file = "VIBE/{}/{}.{}.pkl".format(participant, participant, camera_id)
facebook_output_file = "FB/{}/{}.npy".format(participant, camera_id)
#Video
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
	xs = frame[::3]
	ys = frame[1::3]
	zs = frame[2::3]

# process joints
idx = [0, 8, 10, 12, 7, 3, 5, 1, 4, 6, 2, 11, 13, 9] # change index numbers of vicon joints
# idx = [0, 8, 10, 12, 7, 3, 5, 1, 4, 6, 2, 11, 13, 9, 14]  # change index numbers of vicon joints

joints = np.dstack((xs, ys, zs))[0]
joints = np.insert(joints, 0, (joints[6] + joints[7]) / 2, axis=0)
joints = joints[idx]

####################

#load vicon setup
tree = ET.parse(camera_file)
root = tree.getroot() #cameras

cameras = root.findall('Camera')

camera = cameras[-1]
if(camera_id == "2127314"):
	camera = cameras[-2]  # select camera element

keyframes = camera.find('KeyFrames').findall('KeyFrame')
keyframe = keyframes[0]

#camera calibration
a = float(camera.attrib['PIXEL_ASPECT_RATIO'])
w_0, w_1, w_2 = np.array(keyframe.attrib['VICON_RADIAL2'].split(' ')[3:6], dtype=float)
x_pp, y_pp = np.array(keyframe.attrib['PRINCIPAL_POINT'].split(' '), dtype=float)
x_pp, y_pp = x_pp * 1280 / 1920, y_pp * 1280 / 1920

t = np.array(keyframe.attrib['POSITION'].split(' '), dtype=float)
q = np.array(keyframe.attrib['ORIENTATION'].split(' '), dtype=float)
f = float(keyframe.attrib['FOCAL_LENGTH'])
k = 0 #float(camera.attrib['SKEW'])

R = transforms3d.quaternions.quat2mat(np.roll(q, 1))
P_a = np.concatenate((R, np.expand_dims(np.matmul(R, -t), axis=1)), axis=1)
K = np.array([[f,   k, x_pp], 
            [0, f/a, y_pp],
            [0,   0,    1]])
P = np.matmul(K, P_a)

joints_3d_align = np.matmul(P_a, np.concatenate((joints, np.ones((len(joints), 1))), axis=1).T).T
pelvis = joints_3d_align[0]
joints_3d_relative = joints_3d_align - joints_3d_align[0]

xs = joints_3d_relative[:,0]
ys = joints_3d_relative[:,1]
zs = joints_3d_relative[:,2]

# plotSkeleton(ax, joints_3d_relative, xs, zs, ys, 'grey', 'blue', show_idx=True)

## facebook ##
# fb_joints = np.load(facebook_output_file)
# joints = fb_joints[frame_number]*1000
# joints= np.concatenate((joints[:7, :], joints[10:17, :]))
# idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
# joints = joints[idx]
# xs = joints[:,0]
# ys = joints[:,1]
# zs = joints[:,2]

# mpjpe = mpjpe(torch.from_numpy(joints), torch.from_numpy(joints_3d_relative))
# print(mpjpe)
# # fig, ax = helper.plt_init()
# plotSkeleton(ax, joints, xs, zs, ys, 'black', 'red', show_idx=True)
# plt.show()


# VIBE ###
vibe_joints = joblib.load(vibe_output_file)
m2mm = 1000
vibe_pred = vibe_joints[1]['joints3d'][frame_number] * m2mm

vibe_pred = vibe_pred[25:39]
vibe_pred = np.insert(vibe_pred, 0, (vibe_pred[2] + vibe_pred[3]) / 2, axis=0)
vibe_pred = vibe_pred - vibe_pred[0]
vibe_pred = np.delete(vibe_pred, 13, 0)
idx = [0, 3, 2, 1, 4, 5, 6, 13, 12, 11, 8, 9, 10, 7]  # change index numbers of vicon joints
idx = [0, 3, 2, 1, 4, 5, 6, 13, 12, 11, 10, 9, 8, 7] # 9,11,13 is correct (wrong are 8, 10,12)
idxs = [0, 1,2,3,4,5,6,7,10,9,8,11,12,13] # 9,11,13 is correct (wrong are 8, 10,12)
vibe_pred = vibe_pred[idx]
vibe_pred = vibe_pred[idxs]

xs = vibe_pred[:,0]
ys = vibe_pred[:,1]
zs = vibe_pred[:,2]

mpjpe = mpjpe(torch.from_numpy(vibe_pred), torch.from_numpy(joints_3d_relative))
print(mpjpe)
# fig, ax = helper.plt_init()
plotSkeleton(ax, vibe_pred, xs, zs, ys, 'black', 'red', show_idx=True)
plt.show()