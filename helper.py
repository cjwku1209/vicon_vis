import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xml.etree.ElementTree as ET
import transforms3d
import cv2
import torch


def project2d(joints, setup_path, camera_id):
	# load vicon setup
	tree = ET.parse(setup_path)
	root = tree.getroot()  # cameras

	cameras = root.findall('Camera')

	camera = cameras[-2]
	if(camera_id == 2127314):
		camera = cameras[-1]  # select camera element

	print(camera.attrib['DEVICEID'])

	keyframes = camera.find('KeyFrames').findall('KeyFrame')
	keyframe = keyframes[0]

	# camera calibration
	a = float(camera.attrib['PIXEL_ASPECT_RATIO'])
	w_0, w_1, w_2 = np.array(keyframe.attrib['VICON_RADIAL2'].split(' ')[3:6], dtype=float)
	x_pp, y_pp = np.array(keyframe.attrib['PRINCIPAL_POINT'].split(' '), dtype=float)
	x_pp *= 2/3
	y_pp *= 2/3

	t = np.array(keyframe.attrib['POSITION'].split(' '), dtype=float)
	q = np.array(keyframe.attrib['ORIENTATION'].split(' '), dtype=float)
	f = float(keyframe.attrib['FOCAL_LENGTH'])
	k = 0  # float(camera.attrib['SKEW'])

	R = transforms3d.quaternions.quat2mat(np.roll(q, 1))
	P_a = np.concatenate((R, np.expand_dims(np.matmul(R, -t), axis=1)), axis=1)
	K = np.array([[f, k, x_pp],
					[0, f / a, y_pp],
					[0, 0, 1]])
	P = np.matmul(K, P_a)

	print(P.shape)
	print(joints.shape)
	joints_2d = np.matmul(P, joints.T)
	joints_2d = joints_2d.T

	joints_2d = joints_2d / np.dstack((joints_2d[:, 2], joints_2d[:, 2], joints_2d[:, 2]))[0]

	# print(joints_2d.astype(int))

	# correct radial distortion
	x_r, y_r = joints_2d.T[0], joints_2d.T[1]
	d_x, d_y = x_r - x_pp, a * (y_r - y_pp)
	r = np.sqrt(d_x * d_x + d_y * d_y)
	s = 1 + w_0 * r * r + w_1 * r ** 4
	x_c, y_c = s * d_x + x_pp, (s * d_y + y_pp) / a
	joints_2d_c = np.dstack((x_c, y_c, np.ones(len(x_c))))[0]
	joints_2d = joints_2d_c

	return joints_2d

def to_ndc(joints, setup_path, camera_id):
	# load vicon setup
	tree = ET.parse(setup_path)
	root = tree.getroot()  # cameras

	cameras = root.findall('Camera')

	camera = cameras[-1]
	if(camera_id == 2127314):
		camera = cameras[-2]  # select camera element

	print(camera.attrib['DEVICEID'])

	keyframes = camera.find('KeyFrames').findall('KeyFrame')
	keyframe = keyframes[0]

	# camera calibration
	a = float(camera.attrib['PIXEL_ASPECT_RATIO'])
	w_0, w_1, w_2 = np.array(keyframe.attrib['VICON_RADIAL2'].split(' ')[3:6], dtype=float)
	x_pp, y_pp = np.array(keyframe.attrib['PRINCIPAL_POINT'].split(' '), dtype=float)

	z_ref = joints[2,:]
	size = z_ref * max(width / x_pp, height / y_pp)

		

def preprocess_joints_3d(joints_3d, skip_joints):
	joints_3d = to_matrix(joints_3d, 3)
	# joints_3d = np.array([i for j, i in enumerate(joints_3d) if j not in skip_joints])
	joints_3d = np.hstack((joints_3d, np.ones((joints_3d.shape[0], 1))))
	return joints_3d


def to_matrix(l, n):
	return np.asarray([l[i:i + n] for i in range(0, len(l), n)])


def add_pelvis(joints_2d):
	return np.insert(joints_2d, 0, (joints_2d[7] + joints_2d[6]) / 2, 0)

def draw_joints(img, joints_2d):

	# cv2.line(img, (int(joints_2d[7][0]), int(joints_2d[7][1])), (int(joints_2d[0][0]), int(joints_2d[0][1])),
	#          (255, 0, 255), 2)

	# cv2.line(img, (int(joints_2d[11][0]), int(joints_2d[11][1])), (int(joints_2d[8][0]), int(joints_2d[8][1])),
	#          (255, 0, 255), 2)

	# # Right arm
	# # cv2.line(img, (int(joints_2d[14][0]), int(joints_2d[14][1])), (int(joints_2d[11][0]), int(joints_2d[11][1])),
	# #          (0, 0, 255), 2)
	# cv2.line(img, (int(joints_2d[11][0]), int(joints_2d[11][1])), (int(joints_2d[12][0]), int(joints_2d[12][1])),
	#          (0, 0, 255), 2)
	# cv2.line(img, (int(joints_2d[12][0]), int(joints_2d[12][1])), (int(joints_2d[13][0]), int(joints_2d[13][1])),
	#          (0, 0, 255), 2)

	# # Right Leg
	# cv2.line(img, (int(joints_2d[0][0]), int(joints_2d[0][1])), (int(joints_2d[1][0]), int(joints_2d[1][1])),
	#          (0, 0, 255), 2)
	# cv2.line(img, (int(joints_2d[1][0]), int(joints_2d[1][1])), (int(joints_2d[2][0]), int(joints_2d[2][1])),
	#          (0, 0, 255), 2)
	# cv2.line(img, (int(joints_2d[3][0]), int(joints_2d[3][1])), (int(joints_2d[2][0]), int(joints_2d[2][1])),
	#          (0, 0, 255), 2)
	# cv2.line(img, (int(joints_2d[2][0]), int(joints_2d[2][1])), (int(joints_2d[3][0]), int(joints_2d[3][1])),
	#          (0, 0, 255), 2)

	# # Left Arm
	# # cv2.line(img, (int(joints_2d[14][0]), int(joints_2d[14][1])), (int(joints_2d[8][0]), int(joints_2d[8][1])),
	# #          (255, 0, 0), 2)
	# cv2.line(img, (int(joints_2d[8][0]), int(joints_2d[8][1])), (int(joints_2d[9][0]), int(joints_2d[9][1])),
	#          (255, 0, 0), 2)
	# cv2.line(img, (int(joints_2d[9][0]), int(joints_2d[9][1])), (int(joints_2d[10][0]), int(joints_2d[10][1])),
	#          (255, 0, 0), 2)

	# # Left Leg
	# cv2.line(img, (int(joints_2d[0][0]), int(joints_2d[0][1])), (int(joints_2d[4][0]), int(joints_2d[4][1])),
	#          (255, 0, 0), 2)
	# cv2.line(img, (int(joints_2d[4][0]), int(joints_2d[4][1])), (int(joints_2d[5][0]), int(joints_2d[5][1])),
	#          (255, 0, 0), 2)
	# cv2.line(img, (int(joints_2d[5][0]), int(joints_2d[5][1])), (int(joints_2d[6][0]), int(joints_2d[6][1])),
	#          (255, 0, 0), 2)

	for j in range(joints_2d.shape[0]):
		cv2.circle(img, (int(joints_2d[j][0]), int(joints_2d[j][1])), int(3), (255, 255, 255), 2)
		# if joint_name == None:
		# 	cv2.putText(img, joint_name[j], (int(joints_2d[j][0]), int(joints_2d[j][1])), cv2.FONT_HERSHEY_SIMPLEX,
		# 				0.3, (47, 50, 159), 1, cv2.LINE_AA)
		cv2.putText(img, str(j), (int(joints_2d[j][0]), int(joints_2d[j][1])), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (47, 50, 159), 1, cv2.LINE_AA)
	return img


"""
Mean per-joint position error (i.e. mean Euclidean distance),
often referred to as "Protocol #1" in many papers.
"""
def mpjpe(predicted, target):
	assert predicted.shape == target.shape
	print(torch.mean(abs(predicted - target),1 , True))
	return torch.mean(abs(predicted - target))


def plt_init():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x')
	ax.set_ylabel('z')
	ax.set_zlabel('y')

	return fig, ax


def reshape_axis(xs, ys, zs):
	xs = xs.reshape((14, 1))
	ys = ys.reshape((14, 1))
	zs = zs.reshape((14, 1))
	return xs, ys, zs

def getAxis(data):
	xs = data.narrow(-1, 0, 1).cpu().data.numpy()
	ys = data.narrow(-1, 2, 1).cpu().data.numpy()
	zs = data.narrow(-1, 1, 1).cpu().data.numpy()

	return xs, ys, zs

def plotSkeleton(ax, skel, xs, ys, zs, color1, color2, show_idx=True):
	# Correct aspect ratio (https://stackoverflow.com/a/21765085)
	max_range = np.array([
		xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
	]).max() / 2.0
	mid_x = (xs.max() + xs.min()) * 0.5
	mid_y = (ys.max() + ys.min()) * 0.5
	mid_z = (zs.max() + zs.min()) * 0.5

	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.view_init(elev=20, azim=-100)

	ax.scatter(xs, ys, zs, color=color1)
	
	if (show_idx):
		for i in range(0, 14):
			# ax.text(xs[i][0], ys[i][0], zs[i][0], i)
			ax.text(xs[i], ys[i], zs[i], i)

	joint_connections = [0, 1, 1, 2, 2, 3, 0, 4, 4, 5, 5, 6, 0, 7, 11, 8, 8, 9, 9, 10, 11, 12, 12, 13]
	for i in range(1, 24, 2):
		c = joint_connections[i]
		p = joint_connections[i - 1]
		offset = skel[c] - skel[p]
		ax.quiver(
			[skel[p][0]], [skel[p][2]], [skel[p][1]],
			[offset[0]], [offset[2]], [offset[1]],
			arrow_length_ratio=0.0,
			color=color2,
			alpha=0.5,
		)
	ax.invert_yaxis()

def plotNoConnect(ax, skel, xs, ys, zs, color1, color2, show_idx=True):
	# Correct aspect ratio (https://stackoverflow.com/a/21765085)
	max_range = np.array([
		xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
	]).max() / 2.0
	mid_x = (xs.max() + xs.min()) * 0.5
	mid_y = (ys.max() + ys.min()) * 0.5
	mid_z = (zs.max() + zs.min()) * 0.5

	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.view_init(elev=20, azim=-100)

	ax.scatter(xs, ys, zs, color=color1)

	if (show_idx):
		for i in range(0, xs.size):
			ax.text(xs[i][0], ys[i][0], zs[i][0], i)

	# joint_connections = [16,17,17,13,13,14,14,11,11,10,10,9,9,8,8,3,3,2,2,5,5,6,19,0,19,15,15,18,18,12,19,4,4,7,7,1]
	#
	# for i in range(1, len(joint_connections), 2):
	# 	c = joint_connections[i]
	# 	p = joint_connections[i - 1]
	# 	offset = skel[c] - skel[p]
	# 	ax.quiver(
	# 		[skel[p][0]], [skel[p][2]], [skel[p][1]],
	# 		[offset[0]], [offset[2]], [offset[1]],
	# 		arrow_length_ratio=0.0,
	# 		color=color2,
	# 		alpha=0.5,
	# 	)

def plotOrigVICON(ax, skel, xs, ys, zs, color1, color2, show_idx=True):
	# Correct aspect ratio (https://stackoverflow.com/a/21765085)
	max_range = np.array([
		xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
	]).max() / 2.0
	mid_x = (xs.max() + xs.min()) * 0.5
	mid_y = (ys.max() + ys.min()) * 0.5
	mid_z = (zs.max() + zs.min()) * 0.5

	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.view_init(elev=20, azim=-100)

	ax.scatter(xs, ys, zs, color=color1)

	if (show_idx):
		for i in range(0, xs.size):
			ax.text(xs[i][0], ys[i][0], zs[i][0], i)

	# joint_connections = [16,17,17,13,13,14,14,11,11,10,10,9,9,8,8,3,3,2,2,5,5,6,19,0,19,15,15,18,18,12,19,4,4,7,7,1]
	#
	# for i in range(1, len(joint_connections), 2):
	# 	c = joint_connections[i]
	# 	p = joint_connections[i - 1]
	# 	offset = skel[c] - skel[p]
	# 	ax.quiver(
	# 		[skel[p][0]], [skel[p][2]], [skel[p][1]],
	# 		[offset[0]], [offset[2]], [offset[1]],
	# 		arrow_length_ratio=0.0,
	# 		color=color2,
	# 		alpha=0.5,
	# 	)

def plotOrigVIBE(ax, skel, xs, ys, zs, color1, color2, show_idx=True):
	# Correct aspect ratio (https://stackoverflow.com/a/21765085)
	max_range = np.array([
		xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
	]).max() / 2.0
	mid_x = (xs.max() + xs.min()) * 0.5
	mid_y = (ys.max() + ys.min()) * 0.5
	mid_z = (zs.max() + zs.min()) * 0.5

	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.view_init(elev=20, azim=-100)

	ax.scatter(xs, ys, zs, color=color1)

	if (show_idx):
		for i in range(0, xs.size):
			ax.text(xs[i][0], ys[i][0], zs[i][0], i)

	joint_connections = [0,1,1,2,2,3,3,4,4,5,12,13,12,8,8,7,7,6,12,9,9,10,10,11]

	for i in range(1, len(joint_connections), 2):
		c = joint_connections[i]
		p = joint_connections[i - 1]
		offset = skel[c] - skel[p]
		ax.quiver(
			[skel[p][0]], [skel[p][2]], [skel[p][1]],
			[offset[0]], [offset[2]], [offset[1]],
			arrow_length_ratio=0.0,
			color=color2,
			alpha=0.5,
		)

def plotOrigFB(ax, skel, xs, ys, zs, color1, color2, show_idx=True):
	# Correct aspect ratio (https://stackoverflow.com/a/21765085)
	max_range = np.array([
		xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
	]).max() / 2.0
	mid_x = (xs.max() + xs.min()) * 0.5
	mid_y = (ys.max() + ys.min()) * 0.5
	mid_z = (zs.max() + zs.min()) * 0.5

	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.view_init(elev=20, azim=-100)

	ax.scatter(xs, ys, zs, color=color1)

	if (show_idx):
		for i in range(0, xs.size):
			ax.text(xs[i][0], ys[i][0], zs[i][0], i)

	joint_connections = [3,2,2,1,1,0,0,4,4,5,5,6,0,7,7,8,8,9,9,10,8,14,14,15,15,16,8,11,11,12,12,13]

	for i in range(1, len(joint_connections), 2):
		c = joint_connections[i]
		p = joint_connections[i - 1]
		offset = skel[c] - skel[p]
		ax.quiver(
			[skel[p][0]], [skel[p][2]], [skel[p][1]],
			[offset[0]], [offset[2]], [offset[1]],
			arrow_length_ratio=0.0,
			color=color2,
			alpha=0.5,
		)