import sys
import numpy as np
import numpy.linalg as linalg
import os.path
import cv2 as cv
import platform

OPENPOSE_PATH = '../openpose/python/openpose'
OPENPOSE_MODEL_PATH = '../openpose/models/'

if platform.system() == 'Linux':
    if not os.path.isfile('../openpose/python/openpose/_openpose.so'):
        print('openpen pose is not available: missing _openpose.so', file=sys.stderr)
elif platform.system() == 'Windows':
    if not os.path.isfile('../openpose/python/openpose/_openpose.dll'):
        print('openpen pose is not available: missing _openpose.dll', file=sys.stderr)
else:
    print('not support os', file=sys.stderr)

if not os.path.isfile('../openpose/python/openpose/openpose.py'):
    print('openpen pose is not available: missing openpose.py', file=sys.stderr)

if not os.path.isfile('../openpose/models/pose/body_25/pose_iter_584000.caffemodel'):
    print('missing caffemodel pose_iter_584000.caffemodel', file=sys.stderr)

sys.path.append(OPENPOSE_PATH)
try:
    from openpose import *
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

POSE_BODY_25_PAIRS_RENDER_GPU = (
    1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16,
    16,
    18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24)

POSE_BODY_25_BODY_PARTS = \
    {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "MidHip",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "LBigToe",
        20: "LSmallToe",
        21: "LHeel",
        22: "RBigToe",
        23: "RSmallToe",
        24: "RHeel",
        25: "Background"
    }

POSE_BODY_25_BODY_PART_IDXS = {v: k for k, v in POSE_BODY_25_BODY_PARTS.items()}

KEYPOINT_THRESHOLD = 0.01

def find_pose(img):
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = OPENPOSE_MODEL_PATH
    # Construct OpenPose object allocates GPU memory
    openpose = OpenPose(params)

    keypoints, img_pose = openpose.forward(img, True)

    return keypoints, img_pose

def normalize(vec):
    len = linalg.norm(vec)
    if np.isclose(len, 0.0):
        return vec
    else:
        return (1.0 / len) * vec

def extend_segment(p0, p1, percent):
    dir = p0 - p1
    p0_ = (p0 + percent * dir).astype(np.int32)
    p1_ = (p1 - percent * dir).astype(np.int32)
    len0 = linalg.norm(dir)
    len1 = linalg.norm(p0_ - p1_)
    assert (len1 > len0)
    return p0_, p1_

def normalize(vec):
    len = linalg.norm(vec)
    if np.isclose(len, 0.0):
        return vec
    else:
        return (1.0 / len) * vec

def orthor_dir(vec):
    return np.array([vec[1], -vec[0]])

def find_largest_contour(img_bi, app_type=cv.CHAIN_APPROX_TC89_L1):
    cnt, contours, _ = cv.findContours(img_bi, cv.RETR_LIST, app_type)
    largest_cnt = 0
    largest_area = -1
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt
    return largest_cnt.copy()

from scipy.ndimage import filters
def smooth_contour(contour, sigma=3):
    contour_new = contour.astype(np.float32)
    contour_new[:, 0, 0] = filters.gaussian_filter1d(contour[:, 0, 0], sigma=sigma)
    contour_new[:, 0, 1] = filters.gaussian_filter1d(contour[:, 0, 1], sigma=sigma)
    return contour_new.astype(np.int32)

def contour_length(contour):
    n_point = contour.shape[0]
    l = 0.0
    for i in range(contour.shape[0]):
        i_nxt = (i + 1) % n_point
        l += np.linalg.norm(contour[i,0,:] - contour[i_nxt, 0,:])
    return l

def resample_contour(contour, n_keep_point):
    n_point = contour.shape[0]
    new_contour = np.zeros((n_keep_point, 1, 2), dtype=np.int32)
    cnt_len = contour_length(contour)
    step_len  = cnt_len / float(n_keep_point)
    acc_len = 0.0
    cur_idx = 0
    for i in range(1, n_point):
        p       = contour[i,0,:]
        p_prev  = contour[i-1,0,:]
        cur_e_len = np.linalg.norm(p-p_prev)
        if acc_len + cur_e_len >= step_len:
            residual = cur_e_len - (step_len - acc_len)
            inter_p = p_prev + (1-(residual/cur_e_len))* (p - p_prev)
            new_contour[cur_idx,0,:]  = inter_p
            cur_idx += 1
            acc_len = residual
        else:
            acc_len += cur_e_len

    for i in range(cur_idx, n_keep_point):
        new_contour[i,0,:] = contour[n_point-1, 0, :]

    return new_contour

def int_tuple(vals):
    return tuple(int(v) for v in vals.flatten())

def is_valid_keypoint(keypoint):
    if keypoint[2] < KEYPOINT_THRESHOLD:
        return False
    else:
        return True

def is_valid_keypoint_1(keypoints, name):
    p0 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name]]
    if p0[2] < KEYPOINT_THRESHOLD:
        return False
    else:
        return True

def pair_length(keypoints, name_0, name_1):
    p0 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_0]]
    p1 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_1]]
    if is_valid_keypoint(p0) and is_valid_keypoint(p1):
        return linalg.norm(p0[:2] - p1[:2])
    else:
        return 0

def pair_dir(keypoints, name_0, name_1):
    p0 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_0]]
    p1 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_1]]
    if is_valid_keypoint(p0) and is_valid_keypoint(p1):
        return (p0[:2] - p1[:2])
    else:
        return (0,0)