import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from body_measure import draw_slice_data
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

    data = np.load(data_path)

    contour_f = data.item().get('contour_f')
    contour_s = data.item().get('contour_s')
    slices_f  = data.item().get('slices_f')
    slices_s  = data.item().get('slices_s')
    measure_f = data.item().get('measure_f')
    measure_s = data.item().get('measure_s')

    if contour_f is None or contour_s is None or slices_f is None or slices_s is None:
        print('missing measurement data', file=sys.stderr)
        exit()

    draw_slice_data(img_f, contour_f, slices_f)
    draw_slice_data(img_s, contour_s, slices_s)

    plt.subplot(121)
    plt.imshow(img_f[:,:,::-1])
    plt.subplot(122)
    plt.imshow(img_s[:,:,::-1])
    plt.show()