import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from src.body_measure import draw_slice_data
from src.util import  preprocess_image
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--front_img", required=True, help="front image path")
    ap.add_argument("-s", "--side_img", required=True,   help="side image path")
    ap.add_argument("-d", "--data", required=True,   help="all measurement data including slices, contour, width, height")
    args = vars(ap.parse_args())

    path_f  = args['front_img']
    path_s  = args['side_img']
    data_path    = args['data']

    img_f = cv.imread(path_f)
    if img_f is None:
        print('front image does not exist', file=sys.stderr)
        exit()

    img_s = cv.imread(path_s)
    if img_s is None:
        print('side image does not exist', file=sys.stderr)
        exit()
    print(path_f)
    print(path_s)
    img_f = preprocess_image(img_f)
    img_s = preprocess_image(img_s)

    data = np.load(data_path)

    contour_f = data.item().get('contour_f')
    contour_s = data.item().get('contour_s')
    segments_f  = data.item().get('landmark_segment_f')
    segments_s  = data.item().get('landmark_segment_s')
    seg_dst_f = data.item().get('landmark_segment_dst_f')
    seg_dst_s = data.item().get('landmark_segment_dst_s')
    measurements = data.item().get('measurement')
    segments_height  = data.item().get('landmark_segment_height')

    if contour_f is None or contour_s is None or segments_f is None or segments_s is None:
        print('missing measurement data', file=sys.stderr)
        exit()

    draw_slice_data(img_f, contour_f, segments_f)
    draw_slice_data(img_s, contour_s, segments_s)

    print('length of segment in front and side image\n')
    for id, width in seg_dst_f.items():
        #if id in ['Height', 'CollarBust', 'CollarWaist', 'InsideLeg']:
        #    continue
        if id in seg_dst_s:
            depth = seg_dst_s[id]
        else:
            depth = -1
        print("landmark segment id = {0:30} : width = {1:20}, depth = {2:20}".format(id, width, depth))

    print('\n\n')
    print('segment relative height\n')
    for id, val in segments_height.items():
        print("relative height value of segment {0:30}  = {1:20}".format(id, val))


    print('\n\n')
    print('body measurements in height unit\n')
    for id, val in measurements.items():
        print("measurement type = {0:30} : value = {1:20}".format(id, val))

    #for id, distance in measure_f.items():
    #    if id in ['CollarBust', 'CollarWaist', 'InsideLeg']:
    #        print("slice id = {0:30} : distance = {1:20}".format(id, distance))

    plt.subplot(121)
    plt.imshow(img_f[:,:,::-1])
    plt.subplot(122)
    plt.imshow(img_s[:,:,::-1])
    plt.show()