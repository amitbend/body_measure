import sys
import cv2 as cv
import os
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from pathlib import Path
from openpose_util import is_valid_keypoint, is_valid_keypoint_1, pair_length, pair_dir, find_pose, orthor_dir, extend_segment, find_largest_contour, int_tuple
from openpose_util import  POSE_BODY_25_BODY_PART_IDXS
from pose_to_trimap import gen_fg_bg_masks, head_center_estimate

def extend_rect(rect, percent_w, percent_h):
    w_ext = percent_w * rect[2]
    x = int(rect[0] - 0.5 * w_ext)
    w = int(rect[2] + w_ext)

    h_ext = percent_h * rect[3]
    y = int(rect[1] - 0.5 * h_ext)
    h = int(rect[3] + h_ext)

    return (x, y, w, h)

#rect: x, y, w, h
def grabcut_local_window(img, sil, sure_fg_mask = None, sure_bg_mask = None, rect = None, img_viz = None):
    img_rect   = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    sil_rect   = sil[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    sil_mask_rect = sil_rect > 0

    mask = np.zeros_like(sil_rect, dtype=np.uint8)
    mask[:] = cv.GC_PR_BGD
    mask[sil_mask_rect] = cv.GC_PR_FGD
    if sure_fg_mask is not None:
        sure_fg_mask_rect = sure_fg_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        mask[sure_fg_mask_rect] = cv.GC_FGD
    if sure_bg_mask is not None:
        sure_bg_mask_rect = sure_bg_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        mask[sure_bg_mask_rect] = cv.GC_BGD

    for i in range(2):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv.grabCut(img_rect, mask, None, bgdmodel, fgdmodel, 2, cv.GC_INIT_WITH_MASK)
        sil_1 = np.where((mask == cv.GC_PR_FGD) + (mask == cv.GC_FGD), 255, 0).astype('uint8')

    if img_viz is not None:
        edges = cv.Canny(sil_1, 5, 20)
        edges = cv.morphologyEx(edges, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
        rect_viz = img_viz[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
        rect_viz[edges> 0] = (0,0,255)
        color = np.random.randint(0, 255, 3)
        cv.rectangle(img_viz, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color=int_tuple(color), thickness=3)

    return sil_1

def rect_bounds(points, shape):
    points = np.expand_dims(points, axis=1)
    points = points.astype(np.int32)

    points[:,0,0] = points[:,0,0].clip(min=0, max= shape[1])
    points[:,0,1] = points[:,0,1].clip(min=0, max= shape[0])

    x, y, w, h = cv.boundingRect(points)
    return (x, y, w, h)

def rect_bounds_intersect(rect_0, rect_1):
    x = np.max(rect_0[0], rect_1[0])
    y = np.max(rect_0[1], rect_1[1])

    x_1 = np.min(rect_0[0] + rect_0[2], rect_1[0] + rect_1[2])
    y_1 = np.min(rect_0[1] + rect_0[3], rect_1[1] + rect_1[3])

    return (x, y, x_1 - x, y_1 - y)

def grabcut_local_window_head_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    dir = nose - neck
    over_head = neck + 2 * dir

    points = np.vstack([neck, over_head, lshoulder, rshoulder])
    x, y, w, h = rect_bounds(points, img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part


def grabcut_local_window_shoulder_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]

    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    lshoulder_ext, rshoulder_ext = extend_segment(lshoulder, rshoulder, 0.5)

    neck_ext_0 = neck + 0.5*(nose - neck)
    neck_ext_1 = neck + 0.5*(neck - nose)

    points = np.vstack([lshoulder_ext, rshoulder_ext, neck_ext_0, neck_ext_1])
    x, y, w, h = rect_bounds(points, img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)

    return (x, y, w, h), sil_part

def grabcut_local_window_torso_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    lshoulder_ext, rshoulder_ext = extend_segment(lshoulder, rshoulder, 0.25)

    hip  = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    hip[1] = hip[1] + 0.2 * linalg.norm(lshoulder-rshoulder)

    points = np.vstack([lshoulder_ext, rshoulder_ext, hip])
    x, y, w, h = rect_bounds(points, img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)

    return (x, y, w, h), sil_part

def grabcut_local_window_leg_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    ltoe = keypoints[POSE_BODY_25_BODY_PART_IDXS['LBigToe']][:2]
    rtoe = keypoints[POSE_BODY_25_BODY_PART_IDXS['RBigToe']][:2]

    x, y, w, h = rect_bounds(np.vstack([midhip, ltoe, rtoe]), img.shape)
    x, y, w, h = extend_rect((x,y,w,h), 0.6, 0.3)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def grabcut_local_window_hand_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, left_hand, img_viz):
    if left_hand:
        lshouder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
        lwrist   = keypoints[POSE_BODY_25_BODY_PART_IDXS['LWrist']][:2]
        lwrist   = lwrist + 0.5 * (lwrist - lshouder)
        points   = np.vstack([lshouder, lwrist])
    else:
        rshouder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
        rwrist   = keypoints[POSE_BODY_25_BODY_PART_IDXS['RWrist']][:2]
        rwrist   = rwrist + 0.5 * (rwrist - rshouder)
        points   = np.vstack([rshouder, rwrist])

    x,y,w,h = rect_bounds(points, img.shape)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def refine_silhouette_front_img(img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints, img_viz):
    rect_sils = []

    pair = grabcut_local_window_head_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_shoulder_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_torso_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_leg_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_hand_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, True, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_hand_front_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, False, img_viz)
    rect_sils.append(pair)

    sil_refined = np.zeros_like(sil)
    sil_tmp= np.zeros_like(sil)
    for pair in rect_sils:
        rect = pair[0]
        sil_part = pair[1]
        sil_tmp[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = sil_part
        sil_refined = np.bitwise_or(sil_refined, sil_tmp)

    #sil_refined = np.bitwise_and(sil_refined, sil)
    return sil_refined

def grabcut_local_window_head_side_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    head_center = head_center_estimate(keypoints)
    neck_head = head_center - neck
    over_head = head_center + neck_head
    p0 = head_center + orthor_dir(neck_head)
    p1 = head_center - orthor_dir(neck_head)
    x,y,w,h = rect_bounds(np.vstack([neck, over_head, p0, p1]), img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)

    return (x, y, w, h), sil_part

def grabcut_local_window_torso_side_img(img, sil, sil_bnd_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]

    rect_ext = extend_rect(sil_bnd_rect, 0.4, 0)

    p0 = np.array((rect_ext[0],               neck[1] - 0.05 * sil_bnd_rect[3]))
    p1 = np.array((rect_ext[0] + rect_ext[2], neck[1] - 0.05 * sil_bnd_rect[3]))

    x,y,w,h = rect_bounds(np.vstack([midhip, p0, p1]), img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def grabcut_local_window_hip_thigh_img(img, sil, sil_bnd_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    if is_valid_keypoint_1(keypoints, 'LKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    elif is_valid_keypoint_1(keypoints, 'RKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
    else:
        print('missing keypoints: no knee founded')
        knee = (0.5*img.shape[0], 0.7*img.shape[1])

    rect_ext = extend_rect(sil_bnd_rect, 0.4, 0)
    p0 = np.array((rect_ext[0],               midhip[1] - 0.05 * sil_bnd_rect[3]))
    p1 = np.array((rect_ext[0] + rect_ext[2], midhip[1] - 0.05 * sil_bnd_rect[3]))
    p2 = np.array((rect_ext[0],               knee[1] + 0.05 * sil_bnd_rect[3]))
    p3 = np.array((rect_ext[0] + rect_ext[2], knee[1] + 0.05 * sil_bnd_rect[3]))

    x,y,w,h = rect_bounds(np.vstack([p0, p1, p2, p3]), img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def grabcut_local_window_leg_side_img(img, sil, sil_bnd_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    if is_valid_keypoint_1(keypoints, 'LKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    elif is_valid_keypoint_1(keypoints, 'RKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
    else:
        print('missing keypoints: no knee founded')
        knee = (0.5*img.shape[0], 0.7*img.shape[1])

    rect_ext = extend_rect(sil_bnd_rect, 0.2, 0.05)
    p0 = np.array((rect_ext[0], knee[1] - 0.05*sil_bnd_rect[3]))
    p1 = np.array((rect_ext[0] + rect_ext[2], knee[1]- 0.05*sil_bnd_rect[3]))
    p2 = np.array((rect_ext[0], rect_ext[1]+rect_ext[3]))
    p3 = np.array((rect_ext[0]+rect_ext[2], rect_ext[1]+rect_ext[3]))

    x,y,w,h = rect_bounds(np.vstack([p0, p1, p2, p3]), img.shape)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def refine_silhouette_side_img(img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints, img_viz):
    sil_rect = cv.boundingRect(contour)

    rect_sils = []
    pair = grabcut_local_window_head_side_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_torso_side_img(img, sil, sil_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_leg_side_img(img, sil, sil_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_hip_thigh_img(img, sil, sil_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    sil_refined = np.zeros_like(sil)
    sil_tmp= np.zeros_like(sil)
    for pair in rect_sils:
        rect = pair[0]
        sil_part = pair[1]
        sil_tmp[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = sil_part
        sil_refined = np.bitwise_or(sil_refined, sil_tmp)

    return sil_refined

def load_silhouette(path, img):
    sil = cv.imread(path, cv.IMREAD_GRAYSCALE)
    sil = cv.resize(sil, (img.shape[1], img.shape[0]), cv.INTER_NEAREST)
    ret, sil = cv.threshold(sil, 200, maxval=255, type=cv.THRESH_BINARY)
    return sil

def fix_silhouette(sil):
    sil = cv.morphologyEx(sil, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, ksize=(3,3)))
    return sil

if __name__ == '__main__':
    ROOT_DIR = '/home/khanhhh/data_1/projects/Oh/data/oh_mobile_images/'
    IMG_DIR = f'{ROOT_DIR}images/'
    SILHOUETTE_DIR = f'{ROOT_DIR}silhouette_deeplab/'
    OUT_SILHOUETTE_DIR = f'{ROOT_DIR}silhouette_refined/'

    MARKER_SIZE = 5
    MARKER_THICKNESS = 5
    LINE_THICKNESS = 2

    for f in Path(OUT_SILHOUETTE_DIR).glob('*.*'):
        os.remove(f)

    for img_path in Path(IMG_DIR).glob('*.*'):
        print(img_path)
        if 'side_' in str(img_path):
            fname = img_path.name.replace("side_","")
            is_front_img = False
        elif 'front_' in str(img_path):
            fname = img_path.name.replace("front_","")
            is_front_img = True
        else:
            print('not a front or side image. please attach annation: front_ or side_ to front of the image name')

        img = cv.imread(str(img_path))
        img_org = img.copy()
        keypoints, img_pose = find_pose(img)

        sil = load_silhouette(f'{SILHOUETTE_DIR}{fname}', img)
        sil = fix_silhouette(sil)

        bg_mask = cv.morphologyEx(sil, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (30,30)))

        sure_fg_mask, _ = gen_fg_bg_masks(img, keypoints, front_view=True)
        sure_fg_mask = (sure_fg_mask == 255)
        sure_bg_mask = (bg_mask != 255)

        contour = find_largest_contour(sil, cv.CHAIN_APPROX_TC89_L1)

        sil= cv.morphologyEx(sil, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (20, 20)))
        if is_front_img:
            sil_refined = refine_silhouette_front_img(img_org, sil, sure_fg_mask, sure_bg_mask, contour, keypoints[0, :, :], img)
        else:
            sil_refined = refine_silhouette_side_img(img_org, sil, sure_fg_mask, sure_bg_mask, contour, keypoints[0, :, :], img)

        contour_refined = find_largest_contour(sil_refined, cv.CHAIN_APPROX_TC89_L1)
        sil_final = np.zeros_like(sil)
        cv.fillPoly(sil_final, pts=[contour_refined], color=(255,255,255))
        cv.imwrite(f'{OUT_SILHOUETTE_DIR}{img_path.name}', sil_final)
        continue

        # visualization
        img_1 = img_org.copy()
        cv.drawContours(img_1, [contour], -1, (255, 0, 0), thickness=3)
        cv.drawContours(img_1, [contour_refined], -1, (0, 0, 255), thickness=3)

        plt.subplot(121), plt.imshow(img[:, :, ::-1])
        plt.subplot(122), plt.imshow(img_1[:, :, ::-1])
        #plt.show()
        plt.savefig(f'{OUT_MEASUREMENT_DIR}{img_path.name}', dpi=1000)


