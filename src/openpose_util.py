import sys
import numpy as np
import numpy.linalg as linalg

OPENPOSE_PATH = '/home/khanhhh/data_1/projects/Oh/codes/body_measure/openpose/python/openpose'
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

#OPENPOSE_MODEL_PATH = 'D:\Projects\Oh\\body_measure\openpose\models\\'
OPENPOSE_MODEL_PATH = '/home/khanhhh/data_1/projects/Oh/codes/body_measure/openpose/models/'

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