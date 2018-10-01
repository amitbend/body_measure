import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from src.pose_extract import PoseExtractor
from src.silhouette import SilhouetteExtractor
from src.body_measure import calc_body_slices_util
import time
class BodyMeasure():
    def __init__(self):
        self.pose_extractor = PoseExtractor()

    def process(self, img_f, img_s, height, is_viz_result = True):
        #TODO: smaller image
        img_f = cv.resize(img_f, (1536, 2048), cv.INTER_AREA)
        img_s = cv.resize(img_s, (1536, 2048), cv.INTER_AREA)

        keypoints_f = self.pose_extractor.extract_pose(img_f, debug=False)
        keypoints_s = self.pose_extractor.extract_pose(img_s, debug=False)

        # TODO: Silhouette Deeplab model takes around 10gb of GPU memory.
        # if we construct it in the __init__ function, it will cause out of memory on OpenPose
        sil_extractor = SilhouetteExtractor()
        sil_dl_f, sil_f = sil_extractor.extract_silhouette(img_f, is_front_img=True, keypoints=keypoints_f,
                                                           img_debug=None)
        sil_dl_s, sil_s = sil_extractor.extract_silhouette(img_s, is_front_img=False, keypoints=keypoints_s,
                                                           img_debug=None)
        if is_viz_result == True:
            data, img_viz = calc_body_slices_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height, is_debug=True)
            return data, img_viz
        else:
            data = calc_body_slices_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height, is_debug=False)
            return data

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--front_img", required=True, help="front image path")
    ap.add_argument("-s", "--side_img", required=True,   help="side image path")
    ap.add_argument("-h_cm", "--height_cm", required=True,   help="height in centimet")
    ap.add_argument("-o", "--out_dir", required=True,   help="output directory")
    args = vars(ap.parse_args())

    path_f = Path(args['front_img'])
    path_s = Path(args['side_img'])
    height = args['height_cm']
    OUT_DIR = args['out_dir']
    height = float(height)
    img_f = cv.imread(str(path_f))
    img_s = cv.imread(str(path_s))

    start = time.time()
    body_measure = BodyMeasure()
    #data =  body_measure.process(img_f, img_s, height, is_viz_result=False)
    data, img_viz =  body_measure.process(img_f, img_s, height, is_viz_result=True)
    print(f' total time = {time.time() - start}')

    cv.imwrite(f'{OUT_DIR}/{path_f.stem}.jpg', img_viz)
    np.save(f'{OUT_DIR}/{path_f.stem}.npy', data)
    print(f'output debug result to: {OUT_DIR}/{path_f.stem}.jpg')
    print(f'output slice result to: {OUT_DIR}/{path_f.stem}.npy')

