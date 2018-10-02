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

    img_f = preprocess_image(img_f)
    img_s = preprocess_image(img_s)

    data = np.load(data_path)

    contour_f = data.item().get('contour_f')
    contour_s = data.item().get('contour_s')
    slices_f  = data.item().get('slices_f')
    slices_s  = data.item().get('slices_s')
    measure_f = data.item().get('measure_f')
    measure_s = data.item().get('measure_s')
    measurements = data.item().get('measurements')

    if contour_f is None or contour_s is None or slices_f is None or slices_s is None:
        print('missing measurement data', file=sys.stderr)
        exit()

    draw_slice_data(img_f, contour_f, slices_f)
    draw_slice_data(img_s, contour_s, slices_s)

    print('width and depth of measurement\n')
    for id, width in measure_f.items():
        if id in ['Height', 'CollarBust', 'CollarWaist', 'InsideLeg']:
            continue
        if id in measure_s:
            depth = measure_s[id]
        else:
            depth = -1
        print("slice id = {0:30} : width = {1:20}, depth = {2:20}".format(id, width, depth))

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