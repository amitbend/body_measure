import numpy as np
import cv2 as cv
import platform
import argparse
from pathlib import Path
import os
import sys

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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, help="image folder")
    ap.add_argument("-o", "--output_dir", required=True, help='output pose dir')
    args = vars(ap.parse_args())
    DIR_IN = args['input_dir']
    DIR_OUT = args['output_dir']

    for img_path in Path(DIR_IN).glob('*.*'):
        print(img_path)
        img = cv.imread(str(img_path))
        keypoints, img_pose = find_pose(img)
        cv.imwrite(f'{DIR_OUT}/{img_path.stem}.png',img_pose)
        np.save(f'{DIR_OUT}/{img_path.stem}.npy', keypoints)

