import joblib
import torch
import numpy as np

def load_vibe(file):
	loaded_params = joblib.load(file)
	return loaded_params


def get_vibe_pred(loaded_params, frame_number):
	m2mm = 1000
	vibe_pred = torch.Tensor(loaded_params[1]['joints3d'][frame_number]) * m2mm
	return vibe_pred


def get_translation(gt, vibe_pred):
	pelvis_x, pelvis_z, pelvis_y = gt[0][:-1]
	translation = torch.from_numpy(np.array([pelvis_x, pelvis_y, pelvis_z])) - ((vibe_pred[27] + vibe_pred[[28]]) / 2.0)
	translation = translation.data.numpy()[0]
	return translation


def get_axis(vibe_pred):
	# NOTE: y and z axes are swapped
	xs = vibe_pred.narrow(-1, 0, 1).cpu().data.numpy()
	ys = vibe_pred.narrow(-1, 2, 1).cpu().data.numpy()
	zs = vibe_pred.narrow(-1, 1, 1).cpu().data.numpy()
	zs *= -1

	return xs, ys, zs


def to_tensor(xs, ys, zs):
	pred = []
	for i in range(0, len(xs)):
		pred.append([xs[i], zs[i], ys[i]])
	pred = torch.tensor(pred)
	return pred

def filter_joints(xs, ys, zs, gt, idx):
	xs = xs[25:39]
	ys = ys[25:39]
	zs = zs[25:39]

	pelvis_x, pelvis_z, pelvis_y= gt[0][:-1]
	xs = np.insert(xs, 0, pelvis_x)
	ys = np.insert(ys, 0, pelvis_y)
	zs = np.insert(zs, 0, pelvis_z)

	xs = xs[idx]
	ys = ys[idx]
	zs = zs[idx]

	xs = np.delete(xs, 7)
	ys = np.delete(ys, 7)
	zs = np.delete(zs, 7)

	return xs, ys, zs


def preprocess_vibe(file, frame_number, gt, idx):
	loaded_params = load_vibe(file)
	vibe_pred = get_vibe_pred(loaded_params, frame_number)
	pred_pelvis = (vibe_pred[27] + vibe_pred[[28]]) / 2.0
	vibe_pred -= pred_pelvis #Joints are relative to roots

	translation = get_translation(gt, vibe_pred)
	xs, ys, zs = get_axis(vibe_pred)

	xs += translation[0]
	ys += translation[1]
	zs += translation[2]

	xs, ys, zs = filter_joints(xs, ys, zs, gt, idx)
	pred = to_tensor(xs, ys, zs)
	# idx = [0,4,5,6,1,2,3,7,11,12,13,8,9,10]
	# pred = pred[idx]

	return pred

def original_vibe(file, frame_number):
	loaded_params = load_vibe(file)
	vibe_pred = get_vibe_pred(loaded_params, frame_number)

	xs = vibe_pred[:,0]
	ys = vibe_pred[:, 1]
	zs = vibe_pred[:, 2]

	xs = xs[25:39]
	ys = ys[25:39]
	zs = zs[25:39]

	pred = to_tensor(xs, zs, ys)

	return pred
