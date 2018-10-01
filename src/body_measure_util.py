import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from src.pose_extract import PoseExtractor
from src.silhouette import SilhouetteExtractor
from src.body_measure import calc_body_slices_util
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--front_img", required=True, help="front image path")
    ap.add_argument("-s", "--side_img", required=True,   help="side image path")
    ap.add_argument("-h_cm", "--height_cm", required=True,   help="height in centimet")
    ap.add_argument("-o", "--out_dir", required=True,   help="output directory")
    args = vars(ap.parse_args())

    path_f = args['front_img']
    path_s = args['side_img']
    height = args['height_cm']
    OUT_DIR = args['out_dir']
    height = float(height)
    img_f = cv.imread(path_f)
    img_s = cv.imread(path_s)

    pose_extractor = PoseExtractor()
    #keypoints_s, img_pose_s = pose_extractor.extract_pose(img_s, debug=True)
    #keypoints_f, img_pose_f = pose_extractor.extract_pose(img_f, debug=True)
    keypoints_s = pose_extractor.extract_pose(img_s, debug=False)
    keypoints_f = pose_extractor.extract_pose(img_f, debug=False)
    del pose_extractor

    #Todo: deeplab model takes up almost 10gb GPU memory. needs to find a more efficient way to load it
    sil_extractor = SilhouetteExtractor()

    sil_dl_f, sil_f = sil_extractor.extract_silhouette(img_f, is_front_img = True,  keypoints  = keypoints_f, img_debug=None)
    sil_dl_s, sil_s = sil_extractor.extract_silhouette(img_s, is_front_img = False, keypoints  = keypoints_s, img_debug=None)

    #plt.subplot(121), plt.imshow(img_viz_f[:,:,::-1]), plt.imshow(sil_f, alpha=0.4)
    #plt.subplot(122), plt.imshow(img_viz_s[:,:,::-1]), plt.imshow(sil_s, alpha=0.4)
    #plt.show()

    data, img_viz = calc_body_slices_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height, is_debug=True)

    path_f = Path(path_f)
    path_s = Path(path_s)
    cv.imwrite(f'{OUT_DIR}/{path_f.stem}.jpg', img_viz)
    np.save(f'{OUT_DIR}/{path_f.stem}.npy', data)
    print(f'output debug result to: {OUT_DIR}/{path_f.stem}.jpg')
    print(f'output slice result to: {OUT_DIR}/{path_f.stem}.npy')

