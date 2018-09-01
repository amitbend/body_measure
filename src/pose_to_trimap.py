import sys
import numpy as np
import numpy.linalg as linalg
import cv2 as cv
from openpose_util import POSE_BODY_25_PAIRS_RENDER_GPU, POSE_BODY_25_BODY_PARTS, POSE_BODY_25_BODY_PART_IDXS, KEYPOINT_THRESHOLD
from openpose_util import normalize, extend_segment, find_pose

def is_pair_equal(pair_0, pair_1):
    if (pair_0[0] == pair_1[0] and pair_0[1] == pair_1[1]) or \
            (pair_0[0] == pair_1[1] and pair_0[1] == pair_1[0]):
        return True
    else:
        return False

def is_pair(pair, name_0, name_1):
    if (POSE_BODY_25_BODY_PARTS[pair[0]] == name_0 and POSE_BODY_25_BODY_PARTS[pair[1]] == name_1) or \
            (POSE_BODY_25_BODY_PARTS[pair[1]] == name_0 and POSE_BODY_25_BODY_PARTS[pair[0]] == name_1):
        return True
    else:
        return False

def is_pair_lr(pair, name_0, name_1):
    pair_0_name = POSE_BODY_25_BODY_PARTS[pair[0]]
    pair_1_name = POSE_BODY_25_BODY_PARTS[pair[1]]
    if pair_0_name[0] == 'L' or pair_0_name[0] == 'R':
        pair_0_name = pair_0_name[1:]
    if pair_1_name[0] == 'L' or pair_1_name[0] == 'R':
        pair_1_name = pair_1_name[1:]

    if pair_0_name == name_0 and pair_1_name == name_1 or \
       pair_0_name == name_1 and pair_1_name == name_0:
        return True
    else:
        return False

def is_valid_keypoint(keypoint):
    if keypoint[2] < KEYPOINT_THRESHOLD:
        return False
    else:
        return True

def is_valid_keypoint_1(keypoints, name):
    p0 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name]]
    if p0[2] < KEYPOINT_THRESHOLD:
        return False
    else:
        return True

def pair_length(keypoints, name_0, name_1):
    p0 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_0]]
    p1 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_1]]
    if is_valid_keypoint(p0) and is_valid_keypoint(p1):
        return linalg.norm(p0[:2] - p1[:2])
    else:
        return 0

def pair_dir(keypoints, name_0, name_1):
    p0 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_0]]
    p1 = keypoints[POSE_BODY_25_BODY_PART_IDXS[name_1]]
    if is_valid_keypoint(p0) and is_valid_keypoint(p1):
        return (p0[:2] - p1[:2])
    else:
        return (0,0)

def generate_dilate_width_bg_mask(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    neck_width =  int(0.8 * linalg.norm(neck - nose))

    pairs = POSE_BODY_25_PAIRS_RENDER_GPU

    n_pairs = int(len(pairs) / 2)
    widths = np.zeros(n_pairs, dtype=np.int32)

    for i_pair in range(n_pairs):
        pair = (pairs[i_pair * 2], pairs[i_pair * 2 + 1])
        if is_pair(pair, 'Neck', 'MidHip'):
            widths[i_pair] = int(3 * linalg.norm(neck - nose))
        elif is_pair_lr(pair, 'Hip', 'Knee'):
            widths[i_pair] = int(linalg.norm(neck - nose))
        else:
            widths[i_pair] = int(neck_width)

    return widths

def generate_dilate_width_fg_mask(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    hip_length = pair_length(keypoints, 'LHip','RHip')
    neck_width =  int(0.25 * linalg.norm(neck - nose))
    min_width = 0.2 * neck_width

    pairs = POSE_BODY_25_PAIRS_RENDER_GPU

    n_pairs = int(len(pairs) / 2)
    widths = np.zeros(n_pairs, dtype=np.int32)

    for i_pair in range(n_pairs):
        pair = (pairs[i_pair * 2], pairs[i_pair * 2 + 1])

        if is_pair(pair, 'Neck', 'Nose'):
            widths[i_pair] = int(neck_width)

        elif is_pair(pair, 'Neck', 'MidHip'):
            width = int(hip_length)
            width = max(int(2 * neck_width), width)
            widths[i_pair] = width

        elif is_pair(pair, 'Neck', 'RShoulder'):
            widths[i_pair] = int(1.1 * neck_width)
        elif is_pair(pair, 'RShoulder', 'RElbow'):
            widths[i_pair] = int(0.7 * neck_width)
        elif is_pair(pair, 'RElbow', 'RWrist'):
            widths[i_pair] = int(0.4 * neck_width)

        elif is_pair(pair, 'Neck', 'LShoulder'):
            widths[i_pair] = int(1.1 * neck_width)
        elif is_pair(pair, 'LShoulder', 'LElbow'):
            widths[i_pair] = int(0.7 * neck_width)
        elif is_pair(pair, 'LElbow', 'LWrist'):
            widths[i_pair] = int(0.4 * neck_width)

        elif is_pair(pair, 'MidHip', 'RHip'):
            widths[i_pair] = int(1.2 * neck_width)
        elif is_pair(pair, 'RHip', 'RKnee'):
            widths[i_pair] = int(1.1 * neck_width)
        elif is_pair(pair, 'RKnee', 'RAnkle'):
            widths[i_pair] = int(0.5 * neck_width)
        elif is_pair(pair, 'RAnkle', 'RBigToe'):
            widths[i_pair] = int(0.4 * neck_width)
        elif is_pair(pair, 'RBigToe', 'RSmallToe'):
            widths[i_pair] = int(0.2 * neck_width)

        elif is_pair(pair, 'MidHip', 'LHip'):
            widths[i_pair] = int(1.2 * neck_width)
        elif is_pair(pair, 'LHip', 'LKnee'):
            widths[i_pair] = int(1.1 * neck_width)
        elif is_pair(pair, 'LKnee', 'LAnkle'):
            widths[i_pair] = int(0.5 * neck_width)
        elif is_pair(pair, 'LAnkle', 'LBigToe'):
            widths[i_pair] = int(0.4 * neck_width)
        elif is_pair(pair, 'LBigToe', 'LSmallToe'):
            widths[i_pair] = int(0.2 * neck_width)

        else:
            widths[i_pair] = min_width

    return widths

def draw_bone(img, width, p0, p1, fit=True):
    dir = p0 - p1
    # when width is large, cv.line also extent the segment along the line, which we don't want.
    # so we shrink the segment a bit
    if fit == True and linalg.norm(dir) > 1.3 * width:
        dir = normalize(dir)
        p0 = (p0 - 0.5 * width * dir).astype(np.int32)
        p1 = (p1 + 0.5 * width * dir).astype(np.int32)
    # print(type(width))
    cv.line(img, tuple(p0), tuple(p1), (255, 255, 255), width)

def head_center_estimate(keypoints):
    head_points = []
    #if is_valid_keypoint_1(keypoints, 'Nose'):
    #    head_points.append(keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2])
    if is_valid_keypoint_1(keypoints, 'LEar'):
        head_points.append(keypoints[POSE_BODY_25_BODY_PART_IDXS['LEar']][:2])
    if is_valid_keypoint_1(keypoints, 'REar'):
        head_points.append(keypoints[POSE_BODY_25_BODY_PART_IDXS['REar']][:2])
    #if is_valid_keypoint_1(keypoints, 'LEye'):
    #    head_points.append(keypoints[POSE_BODY_25_BODY_PART_IDXS['LEye']][:2])
    #if is_valid_keypoint_1(keypoints, 'REye'):
    #    head_points.append(keypoints[POSE_BODY_25_BODY_PART_IDXS['REye']][:2])

    return np.mean(np.array(head_points), axis=0)

def generate_fg_mask(img, keypoints):
    n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU) / 2)
    fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    bone_widths = generate_dilate_width_fg_mask(keypoints)

    for i_pair in range(n_pairs):
        idx_0 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2]
        idx_1 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2 + 1]

        #print(f'{POSE_BODY_25_BODY_PARTS[idx_0]}:{keypoints[idx_0, :][2]} -- {POSE_BODY_25_BODY_PARTS[idx_1]}:{keypoints[idx_1, :][2]}')
        #print(f'{keypoints[idx_0, :]} - {keypoints[idx_1, :]}')
        if not is_valid_keypoint(keypoints[idx_0, :]) or not is_valid_keypoint(keypoints[idx_1, :]):
            continue

        # process them later
        if POSE_BODY_25_BODY_PARTS[idx_0] == 'Nose' or POSE_BODY_25_BODY_PARTS[idx_1] == 'Nose':
            continue

        kpoint_0 = keypoints[idx_0, :2].astype(np.int32)
        kpoint_1 = keypoints[idx_1, :2].astype(np.int32)

        if is_pair((idx_0, idx_1), 'Neck', 'MidHip'):
            kpoint_0, kpoint_1 = extend_segment(kpoint_0, kpoint_1, 0.1)

        draw_bone(fg_mask, bone_widths[i_pair], kpoint_0, kpoint_1)

    neck =  keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    head_center = head_center_estimate(keypoints)
    neck_len = linalg.norm(neck-head_center)

    draw_bone(fg_mask, int(0.2*neck_len), neck, head_center)
    cv.circle(fg_mask, tuple(head_center), int(0.3*neck_len), (255,255,255), thickness=cv.FILLED)

    return fg_mask

def generate_bg_mask(img, keypoints):
    n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU) / 2)

    bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    dilate_widths = generate_dilate_width_bg_mask(keypoints)

    for i_pair in range(n_pairs):
        idx_0 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2]
        idx_1 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2 + 1]

        if not is_valid_keypoint(keypoints[idx_0, :]) or not is_valid_keypoint(keypoints[idx_1, :]):
            continue

        # process them later
        if POSE_BODY_25_BODY_PARTS[idx_0] == 'Nose' or POSE_BODY_25_BODY_PARTS[idx_1] == 'Nose':
            continue

        kpoint_0 = keypoints[idx_0, :2].astype(np.int32)
        kpoint_1 = keypoints[idx_1, :2].astype(np.int32)

        if is_pair((idx_0, idx_1), 'Neck', 'Nose') or \
                is_pair((idx_0, idx_1), 'LElbow', 'LWrist') or is_pair((idx_0, idx_1), 'RElbow', 'RWrist'):
            kpoint_0, kpoint_1 = extend_segment(kpoint_0, kpoint_1, 0.9)

        elif is_pair_lr((idx_0, idx_1), 'Knee', 'Ankle') or \
             is_pair_lr((idx_0, idx_1), 'Hip',  'Knee'):
            kpoint_0, kpoint_1 = extend_segment(kpoint_0, kpoint_1, 0.2)

        elif is_pair((idx_0, idx_1), 'Neck', 'MidHip'):
            kpoint_0, kpoint_1 = extend_segment(kpoint_0, kpoint_1, 0.6)

        draw_bone(bg_mask, dilate_widths[i_pair], kpoint_0, kpoint_1)

    # here we approximate neck dir by the direction from mid hip to neck
    # because in side view, direction from neck to nose doesn't conincide with neck
    neck =  keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    head_center = head_center_estimate(keypoints)
    neck_len = linalg.norm(neck-head_center)
    draw_bone(bg_mask, int(0.5*neck_len), neck, head_center)
    cv.circle(bg_mask, tuple(head_center), int(neck_len), (255,255,255), thickness=cv.FILLED)

    #bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (50, 50)), iterations=4)

    return 255 - bg_mask

def gen_fg_bg_masks(img, keypoints, front_view = True):
    if keypoints.shape[0] < 1:
        fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        fg_mask = generate_fg_mask(img, keypoints[0,:,:])

        bg_mask = generate_bg_mask(img, keypoints[0,:,:])

        amap = np.zeros(fg_mask.shape, dtype=np.uint8)
        amap[fg_mask > 0] = 255
        amap[np.bitwise_and(np.bitwise_not(bg_mask > 0), np.bitwise_not(fg_mask > 0))] = 155

    # cv.imwrite(f'{OUT_DIR}{Path(img_path).name}', output_image)
    # cv.imwrite(f'{OUT_DIR_ALPHA_MAP}{Path(img_path).name}', amap)
    return fg_mask, bg_mask

from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    DIR_ROOT = 'D:/Projects/Oh/data/images/mobile/oh_images/'
    DIR_IMG = f'{DIR_ROOT}/images/'
    DIR_TRIMAP = f'{DIR_ROOT}/tri_map/'

    for path in Path(DIR_IMG).glob('*.*'):
        #if 'IMG_1942' not in str(path):
        #     continue
        img = cv.imread(str(path))

        keypoints, img_pose = find_pose(img)
        fg_mask, bg_mask = gen_fg_bg_masks(img, keypoints)
        tri_map = np.zeros(fg_mask.shape, dtype=np.uint8)
        tri_map[bg_mask!=255] = 125
        tri_map[fg_mask==255] = 255
        tri_map = cv.resize(tri_map, (240,320), interpolation=cv.INTER_NEAREST)

        cv.imwrite(f'{DIR_TRIMAP}{path.name}', tri_map)

        # plt.subplot(121), plt.imshow(img_pose[:,:,::-1])
        # plt.subplot(122)
        # plt.imshow(img[:,:,::-1])
        # plt.imshow(fg_mask, alpha=0.5)
        # plt.imshow(bg_mask, alpha=0.5)
        # plt.show()
        #plt.savefig(f'{DIR_ROOT}debug/{path.name}', dpi=500)
