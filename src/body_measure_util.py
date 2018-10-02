import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from src.pose_extract import PoseExtractor
from src.silhouette import SilhouetteExtractor
from src.body_measure import calc_body_landmarks_util
from src.util import preprocess_image
import time
class BodyMeasure():
    def __init__(self):
        self.pose_extractor = PoseExtractor()
        pass

    def process(self, img_f, img_s, height, is_viz_result = True):

        img_f = preprocess_image(img_f)
        img_s = preprocess_image(img_s)

        keypoints_f = self.pose_extractor.extract_pose(img_f, debug=False)
        keypoints_s = self.pose_extractor.extract_pose(img_s, debug=False)

        sil_extractor = SilhouetteExtractor()
        # TODO: Silhouette Deeplab model takes around 10gb of GPU memory.
        # if we construct it in the __init__ function, it will cause out of memory on OpenPose
        _, sil_f = sil_extractor.extract_silhouette(img_f, is_front_img=True,  keypoints=keypoints_f,
                                                           img_debug=None)
        _, sil_s = sil_extractor.extract_silhouette(img_s, is_front_img=False, keypoints=keypoints_s,
                                                           img_debug=None)
        del sil_extractor

        if is_viz_result == True:
            data, img_viz   = calc_body_landmarks_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height, is_debug=True)
            return data, img_viz
        else:
            data            = calc_body_landmarks_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height, is_debug=False)
            return data

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--front_img", required=True, help="front image path")
    ap.add_argument("-s", "--side_img", required=True,   help="side image path")
    ap.add_argument("-h_cm", "--height_cm", required=True,   help="height in centimet")
    ap.add_argument("-o", "--out_dir", required=True,   help="output directory")
    ap.add_argument("-d", "--debug", required=False,   default=1, help="set it to 1 or 0 to output visualization")
    args = vars(ap.parse_args())

    path_f = Path(args['front_img'])
    path_s = Path(args['side_img'])
    height = args['height_cm']
    OUT_DIR = args['out_dir']
    is_debug = int(args['debug']) > 0

    height = float(height)
    img_f = cv.imread(str(path_f))
    img_s = cv.imread(str(path_s))

    body_measure = BodyMeasure()
    start = time.time()
    #data =  body_measure.process(img_f, img_s, height, is_viz_result=False)
    data, img_viz =  body_measure.process(img_f, img_s, height, is_viz_result=True)
    print(f' total time = {time.time() - start}')

    np.save(f'{OUT_DIR}/{path_f.stem}.npy', data)
    print(f'output slice result to: {OUT_DIR}/{path_f.stem}.npy')

    if is_debug:
        cv.imwrite(f'{OUT_DIR}/{path_f.stem}.jpg', img_viz)
        print(f'output debug result to: {OUT_DIR}/{path_f.stem}.jpg')

