import numpy as np
import cv2 as cv
from pathlib import Path
import argparse
from openpose_util import find_pose

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, help="image folder")
    ap.add_argument("-o", "--output_dir", required=True, help='output pose dir')
    args = vars(ap.parse_args())
    DIR_IN = args['input_dir']
    DIR_OUT = args['output_dir']

    for img_path in Path(DIR_IN).glob('*.*'):
        img = cv.imread(str(img_path))
        keypoints, img_pose = find_pose(img)
        cv.imwrite(f'{DIR_OUT}/{img_path.stem}.png',img_pose)
        np.save(f'{DIR_OUT}/{img_path.stem}.npy', keypoints)

