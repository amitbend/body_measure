import cv2 as cv
import os
import tensorflow.test as test_util
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
import argparse
from src.pose_extract import PoseExtractor
from src.silhouette import SilhouetteExtractor
from src.body_measure import calc_body_landmarks_util
from src.util import preprocess_image

class BodyMeasure():
    def __init__(self):
        # use this option if you have a good GPU because tensorflow is very hungry for memory. it starts with 1GB and grows quickly to an upper limit that is configued by us
        self.use_gpu = False
        #we need more checking to be able to use deeplab mobile version. its precision is not good now
        self.use_deeplab_mobile=False
        #TODO: use device_lib of tensorfow will cause an allocation of 10GB. we need a smarter way to do it
        #self.check_hardware_requirement()
        mem_require = self.total_min_gpu_mem_requirement()
        print(f'Please make sure that you have at least {mem_require} GB of free GPU mem')
        self.pose_extractor = PoseExtractor()
        self.sil_extractor = SilhouetteExtractor(use_gpu=self.use_gpu, use_mobile_model=self.use_deeplab_mobile)
        pass

    def get_available_gpus(self):
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [d for d in local_device_protos if d.device_type == 'GPU']

    #require mininum memory requiremetn in gigabite unit
    def min_deeplab_mem_requirement(self):
        if self.use_gpu:
            if self.use_deeplab_mobile:
                return 0.5
            else:
                return 1.3
        else:
            #don't know why. even we don't construct tensorflow with GPU, the GPU mem still increases by 0.4G
            return 0.4

    #for openpose, we always use GPU. not CPU support now
    def min_openpose_mem_requirement(self):
        return 1.3

    def total_min_gpu_mem_requirement(self):
        return self.min_deeplab_mem_requirement() + self.min_openpose_mem_requirement()

    def check_hardware_requirement(self):
        if not test_util.is_gpu_available() and self.use_gpu:
            print('GPU is not available but Deeplab is configued with GPU', file=sys.stderr)

        gpus = self.get_available_gpus()
        for gpu in gpus:
            print(gpu)

    def process(self, img_f, img_s, height, is_viz_result = True):

        img_f = preprocess_image(img_f)
        img_s = preprocess_image(img_s)

        keypoints_f = self.pose_extractor.extract_pose(img_f, debug=False)
        keypoints_s = self.pose_extractor.extract_pose(img_s, debug=False)

        # TODO: Silhouette Deeplab model takes around 10gb of GPU memory.
        # if we construct it in the __init__ function, it will cause out of memory on OpenPose
        _, sil_f = self.sil_extractor.extract_silhouette(img_f, is_front_img=True,  keypoints=keypoints_f,
                                                           img_debug=None)
        _, sil_s = self.sil_extractor.extract_silhouette(img_s, is_front_img=False, keypoints=keypoints_s,
                                                           img_debug=None)

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
    n_test = 1
    for i in range(n_test):
        start = time.time()
        #data =  body_measure.process(img_f, img_s, height, is_viz_result=False)
        data, img_viz =  body_measure.process(img_f, img_s, height, is_viz_result=True)
        print(f'total time of test {i} = {time.time() - start}')

    np.save(f'{OUT_DIR}/{path_f.stem}.npy', data)
    print(f'output slice result to: {OUT_DIR}/{path_f.stem}.npy')

    if is_debug:
        cv.imwrite(f'{OUT_DIR}/{path_f.stem}.jpg', img_viz)
        print(f'output debug result to: {OUT_DIR}/{path_f.stem}.jpg')

    #time.sleep(10000)
