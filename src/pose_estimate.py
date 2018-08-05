import sys
import cv2 as cv
import os
from sys import platform
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from pathlib import Path

from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points

OPENPOSE_PATH = 'D:\Projects\Oh\\body_measure\openpose\\build\python\openpose'
sys.path.append(OPENPOSE_PATH)
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

POSE_BODY_25_PAIRS_RENDER_GPU = (1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   14,19,19,20,14,21, 11,22,22,23,11,24)
POSE_BODY_25_BODY_PARTS = \
{
    0 : "Nose",
    1 : "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel",
    25: "Background"
}

POSE_BODY_25_BODY_PART_IDXS = {v:k for k,v in POSE_BODY_25_BODY_PARTS.items()}

KEYPOINT_THRESHOLD = 0.01

def is_pair_equal(pair_0, pair_1):
    if (pair_0[0] == pair_1[0] and pair_0[1] == pair_1[1])  or \
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

def normalize(vec):
    len = linalg.norm(vec)
    if np.isclose(len, 0.0):
        return vec
    else:
        return (1.0/len) * vec

def extend_rect(p0, p1, extent):
    dir = p0 - p1
    dir = normalize(dir)
    n = np.array([dir[1], -dir[0]])
    c0 = (p0 + 0.5 * extent * n).astype(np.int32)
    c1 = (p0 - 0.5 * extent * n).astype(np.int32)
    c2 = (p1 - 0.5 * extent * n).astype(np.int32)
    c3 = (p1 + 0.5 * extent * n).astype(np.int32)
    return np.array([c0, c1, c2, c3])

def generate_bone_width():
    neck_mid_hip = 200
    neck_nose    = 50

    min_width = 5

    pairs = POSE_BODY_25_PAIRS_RENDER_GPU

    n_pairs = int(len(pairs)/2)
    widths  = np.zeros(n_pairs, dtype=np.int32)

    for i_pair in range(n_pairs):
        pair = (pairs[i_pair*2], pairs[i_pair*2+1])

        if is_pair(pair, 'Neck', 'Nose'):
            widths[i_pair] = int(neck_nose)

        elif is_pair(pair, 'Neck', 'MidHip'):
            widths[i_pair] = int(neck_mid_hip)

        elif is_pair(pair, 'Neck', 'RShoulder'):
            widths[i_pair] = int(1.1 * neck_nose)
        elif is_pair(pair, 'RShoulder', 'RElbow'):
            widths[i_pair] = int(0.7 * neck_nose)
        elif is_pair(pair, 'RElbow', 'RWrist'):
            widths[i_pair] = int(0.4 * neck_nose)

        elif is_pair(pair, 'Neck', 'LShoulder'):
            widths[i_pair] = int(1.1 * neck_nose)
        elif is_pair(pair, 'LShoulder', 'LElbow'):
            widths[i_pair] = int(0.7 * neck_nose)
        elif is_pair(pair, 'LElbow', 'LWrist'):
            widths[i_pair] = int(0.4 * neck_nose)

        elif is_pair(pair, 'MidHip', 'RHip'):
            widths[i_pair] = int(1.2 * neck_nose)
        elif is_pair(pair, 'RHip', 'RKnee'):
            widths[i_pair] = int(1.1 * neck_nose)
        elif is_pair(pair, 'RKnee', 'RAnkle'):
            widths[i_pair] = int(1.0 * neck_nose)
        elif is_pair(pair, 'RAnkle', 'RBigToe'):
            widths[i_pair] = int(0.4 * neck_nose)
        elif is_pair(pair, 'RBigToe', 'RSmallToe'):
            widths[i_pair] = int(0.2 * neck_nose)

        elif is_pair(pair, 'MidHip', 'LHip'):
            widths[i_pair] = int(1.2 * neck_nose)
        elif is_pair(pair, 'LHip', 'LKnee'):
            widths[i_pair] = int(1.1 * neck_nose)
        elif is_pair(pair, 'LKnee', 'LAnkle'):
            widths[i_pair] = int(1.0 * neck_nose)
        elif is_pair(pair, 'LAnkle', 'LBigToe'):
            widths[i_pair] = int(0.4 * neck_nose)
        elif is_pair(pair, 'LBigToe', 'LSmallToe'):
            widths[i_pair] = int(0.2 * neck_nose)

        else:
            widths[i_pair] =  min_width

    return widths

def draw_bone(img, width, p0, p1, fit = True):
    dir = p0 - p1
    #when width is large, cv.line also extent the segment along the line, which we don't want.
    #so we shrink the segment a bit
    if fit == True and linalg.norm(dir) > 1.3 * width:
        dir = normalize(dir)
        p0 = (p0 - 0.5 * width * dir).astype(np.int32)
        p1 = (p1 + 0.5 * width * dir).astype(np.int32)
    #print(type(width))
    cv.line(img, tuple(p0), tuple(p1), (255, 255, 255), width)

def is_valid_keypoint(keypoint):
    if keypoint[2] < KEYPOINT_THRESHOLD or keypoint[2] < KEYPOINT_THRESHOLD:
        return False
    else:
        return True

def generate_fg_mask(img, keypoints, bone_widths):
    n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU)/2)
    fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i_pair in range(n_pairs):
        idx_0 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2]
        idx_1 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2 + 1]

        if not is_valid_keypoint(keypoints[0, idx_0, :]) or not is_valid_keypoint(keypoints[0, idx_1, :]):
            continue

        kpoint_0 = keypoints[0, idx_0, :2].astype(np.int32)
        kpoint_1 = keypoints[0, idx_1, :2].astype(np.int32)

        draw_bone(fg_mask, bone_widths[i_pair], kpoint_0, kpoint_1)

    return fg_mask

def extend_segment(p0, p1, percent):
   dir = p0 - p1
   p0_  = (p0 + percent * dir).astype(np.int32)
   p1_  = (p1 - percent * dir).astype(np.int32)
   len0 = linalg.norm(dir)
   len1 = linalg.norm(p0_ - p1_)
   assert (len1 > len0)
   return p0_, p1_

def generate_bg_mask(img, keypoints, bone_widths):
    n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU)/2)

    bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    bone_widths_dilated = (3 * bone_widths).astype(np.int32)

    for i_pair in range(n_pairs):
        idx_0 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2]
        idx_1 = POSE_BODY_25_PAIRS_RENDER_GPU[i_pair * 2 + 1]

        if not is_valid_keypoint(keypoints[0, idx_0, :]) or not is_valid_keypoint(keypoints[0, idx_1, :]):
            continue

        kpoint_0 = keypoints[0, idx_0, :2].astype(np.int32)
        kpoint_1 = keypoints[0, idx_1, :2].astype(np.int32)

        if is_pair((idx_0, idx_1), 'Neck', 'Nose') or \
            is_pair((idx_0, idx_1), 'Neck', 'Nose') or \
            is_pair((idx_0, idx_1), 'LElbow', 'LWrist') or \
            is_pair((idx_0, idx_1), 'RElbow', 'RWrist'):
            kpoint_0, kpoint_1 = extend_segment(kpoint_0, kpoint_1, 0.9)

        if is_pair((idx_0, idx_1), 'Neck', 'MidHip'):
            kpoint_0, kpoint_1 = extend_segment(kpoint_0, kpoint_1, 0.6)

        draw_bone(bg_mask, bone_widths_dilated[i_pair], kpoint_0, kpoint_1)

    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (50,50)), iterations=4)

    return 255 - bg_mask

def gen_fg_bg_masks(img, keypoints, bontwidths):
    if keypoints.shape[0] < 1:
        fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        fg_mask = generate_fg_mask(img, keypoints, bone_widths)
        bg_mask = generate_bg_mask(img, keypoints, bone_widths)

        amap = np.zeros(fg_mask.shape, dtype=np.uint8)
        amap[fg_mask > 0] = 255
        amap[np.bitwise_and(np.bitwise_not(bg_mask > 0), np.bitwise_not(fg_mask > 0))] = 155

    #cv.imwrite(f'{OUT_DIR}{Path(img_path).name}', output_image)
    #cv.imwrite(f'{OUT_DIR_ALPHA_MAP}{Path(img_path).name}', amap)
    return fg_mask, bg_mask

def find_largest_contour(img_bi):
    cnt, contours, _ = cv.findContours(img_bi, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    largest_cnt = 0
    largest_area = -1
    for cnt in contours:
        area =cv.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt
    return largest_cnt.copy()

def closest_point_segment(p0, p1, p):
    p10 = p1 - p0
    a = np.dot(p, p10) - np.dot(p0, p10)
    b = np.dot(p10, p10)
    t = a/b
    if t  > 1.0:
        return p1
    elif t < -1.0:
        return p0
    else:
        return p0 + t *p10

def dst_point_segment(p0, p1, p):
    tmp = closest_point_segment(p0, p1, p)
    tmp = p - tmp
    return np.norm(tmp)

def closest_point_contour_points(contour, point):
    cls_idx = 0
    p = contour[cls_idx,:,:]
    cls_dst = np.linalg.norm(p-point)
    diffs = contour - point
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    return np.argmin(dists)

import sys
def closest_point_contour_segments(contour, point):
    contour = contour.astype(np.float32)
    point = point.astype(np.float32)
    cls_dst = sys.float_info.max
    cls_p = None
    n_point = len(contour)
    for i in range(n_point):
        p0 = contour[i,:,:].flatten()
        p1 = contour[(i+1)%n_point,:,:].flatten()
        tmp_p = closest_point_segment(p0, p1, point)
        dst =  np.linalg.norm(tmp_p - point)
        if dst < cls_dst:
            cls_dst = dst
            cls_p = tmp_p

    return cls_p

def closest_point_points(point, points):
    g0 = Point(point)
    g1 = MultiPoint(points)
    closest = nearest_points(g0, g1)[1]
    return np.array([closest.x, closest.y])

def isect_segment_contour(contour, p0, p1):
    a = LineString([p0, p1])
    cnt_points = [contour[i].flatten() for i in range(len(contour))]
    b = LineString(cnt_points)
    ipoints = a.intersection(b)
    return [(p.x, p.y) for p in ipoints]

def find_symmetric_keypoints_on_boundary(contour, keypoints, keypoint_name):
    left_name  = "".join(("L", keypoint_name))
    right_name = "".join(("R",keypoint_name))
    left = keypoints[POSE_BODY_25_BODY_PART_IDXS[left_name]][:2]
    right = keypoints[POSE_BODY_25_BODY_PART_IDXS[right_name]][:2]
    left_bdr = closest_point_contour_segments(contour, left)
    right_bdr = closest_point_contour_segments(contour, right)
    return left_bdr, right_bdr

def extend_segments(p0, p1, percent):
    dir = p0 - p1
    p0_ext = p0 + percent * dir
    p1_ext = p1 - percent * dir
    return p0_ext, p1_ext

def measure_acromial_height(contour, keypoints, left = True):
    if left:
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    else:
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

    lAnkle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rAnkle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    lAnkle_ext, rAnkle_ext = extend_segment(lAnkle, rAnkle, 3)

    on_base_line =  nearest_points(LineString([lAnkle_ext, rAnkle_ext]), Point(shoulder))[0]
    on_shoulder  = nearest_points(LineString([contour[i].flatten() for i in range(len(contour))]), Point(shoulder))[0]

    return [(on_base_line.x, on_base_line.y), (on_shoulder.x, on_shoulder.y)]

def axis_front_view(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    axis = neck - midhip
    axis = normalize(axis)
    return axis

def axis_side_view(keypoints):
    neck   = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    lankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    ankle = 0.5 * (lankle + rankle)
    axis = neck - ankle
    axis = normalize(axis)
    return axis

def orthor_dir(vec):
    return np.array([vec[1], -vec[0]])

def neck_hip_length(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    hip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    return np.linalg.norm(neck-hip)

def estimate_front_hip_bdr_points(contour_front, keypoints_front):
    axis_front = axis_front_view(keypoints_front)
    lhip_front = keypoints_front[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip_front = keypoints_front[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    lhip_front_ext, rhip_front_ext = extend_segment(lhip_front, rhip_front, 4)
    ipoints = isect_segment_contour(contour_front, lhip_front_ext, rhip_front_ext)
    hip_bdr_front_0 = closest_point_points(lhip_front, ipoints)
    hip_bdr_front_1 = closest_point_points(rhip_front, ipoints)
    return np.vstack([hip_bdr_front_0, hip_bdr_front_1])

def estimate_side_hip_bdr_points(contour_side, keypoints_side):
    axis_side = axis_front_view(keypoints_side)
    axis_ortho = orthor_dir(axis_side)
    midhip_side = keypoints_side[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    ref_length = neck_hip_length(keypoints_side)
    s0 = midhip_side + axis_ortho * 0.5*ref_length
    s1 = midhip_side - axis_ortho * 0.5*ref_length
    ipoints = isect_segment_contour(contour_side, s0, s1)
    hip_bdr_side_0 = closest_point_points(s0, ipoints)
    hip_bdr_side_1 = closest_point_points(s1, ipoints)
    return np.vstack([hip_bdr_side_0, hip_bdr_side_1])

def estimate_front_waist_bdr_points(contour, keypoints):
    axis = axis_front_view(keypoints)
    axis_ortho = orthor_dir(axis)

    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    neck   = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    dir = neck - midhip
    waist_center =  midhip + 0.333 * dir

    ref_length = neck_hip_length(keypoints)
    s0 = waist_center + axis_ortho * 0.5*ref_length
    s1 = waist_center - axis_ortho * 0.5*ref_length

    ipoints = isect_segment_contour(contour, s0, s1)

    waist_bdr_front_0 = closest_point_points(s0, ipoints)
    waist_bdr_front_1= closest_point_points(s1, ipoints)

    return np.vstack([waist_bdr_front_0, waist_bdr_front_1])

def estimate_side_waist_bdr_points(contour, keypoints):
    axis = axis_side_view(keypoints)
    axis_ortho = orthor_dir(axis)

    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    neck   = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    dir = neck - midhip
    waist_center =  midhip + 0.333 * dir

    ref_length = neck_hip_length(keypoints)
    s0 = waist_center + axis_ortho * 0.5*ref_length
    s1 = waist_center - axis_ortho * 0.5*ref_length

    ipoints = isect_segment_contour(contour, s0, s1)

    waist_bdr_side_0 = closest_point_points(s0, ipoints)
    waist_bdr_side_1 = closest_point_points(s1, ipoints)

    return np.vstack([waist_bdr_side_0, waist_bdr_side_1])

def estimate_front_neck_brd_points(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]

    dir = nose - neck
    len = np.linalg.norm(dir)
    pos = neck + 0.4 * dir

    dir = normalize(dir)
    dir_ortho = orthor_dir(dir)
    p0 = pos + dir_ortho * len
    p1 = pos - dir_ortho * len

    ipoints = isect_segment_contour(contour, p0, p1)

    neck_bdr_front_0 = closest_point_points(p0, ipoints)
    neck_bdr_front_1 = closest_point_points(p1, ipoints)

    return np.vstack([neck_bdr_front_0, neck_bdr_front_1])

def estimate_side_neck_brd_points(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]

    dir = nose - neck
    len = np.linalg.norm(dir)
    pos = neck + 0.4 * dir

    dir = normalize(dir)
    dir_ortho = orthor_dir(dir)
    p0 = pos + dir_ortho * len
    p1 = pos - dir_ortho * len

    ipoints = isect_segment_contour(contour, p0, p1)

    neck_bdr_side_0 = closest_point_points(p0, ipoints)
    neck_bdr_side_1 = closest_point_points(p1, ipoints)

    return np.vstack([neck_bdr_side_0, neck_bdr_side_1])

def estimate_front_inside_leg_bdr_points(contour, keypoints):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    lankle  = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rankle  = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    ankle = 0.5 * (lankle + rankle)
    return np.vstack([midhip, ankle])

def estimate_side_inside_leg_bdr_points(contour, keypoints):
    return estimate_front_inside_leg_bdr_points(contour, keypoints)

def estimate_front_shoulder_points(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]

    dir = nose - neck
    len = np.linalg.norm(dir)
    pos = neck + 0.1 * dir

    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    dir_ortho = lshoulder - rshoulder
    p0 = pos + dir_ortho * len
    p1 = pos - dir_ortho * len

    ipoints = isect_segment_contour(contour, p0, p1)

    p0 = closest_point_points(p0, ipoints)
    p1= closest_point_points(p1, ipoints)

    return np.vstack([p0, p1])

def estimate_front_arm(contour, keypoints):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    lwrist    = keypoints[POSE_BODY_25_BODY_PART_IDXS['LWrist']][:2]
    return np.vstack([lshoulder, lwrist])

def estimate_side_arm(contour, keypoints):
    return estimate_front_arm(contour, keypoints)

def estimate_front_around_arm(contour, keypoints):
    lelbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]

    lshoulder   = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    lwrist      = keypoints[POSE_BODY_25_BODY_PART_IDXS['LWrist']][:2]
    dir = lshoulder - lwrist
    len_ref = np.linalg.norm(dir)
    dir = normalize(dir)
    dir_ortho = orthor_dir(dir)

    p0 = lelbow + 0.2*len_ref * dir_ortho
    p1 = lelbow - 0.2*len_ref * dir_ortho
    ipoints = isect_segment_contour(contour, p0, p1)
    p0 = closest_point_points(p0, ipoints)
    p1= closest_point_points(p1, ipoints)

    return np.vstack([p0, p1])

def visualize_measusements(img, sil, keypoints):
    contour = find_largest_contour(sil)
    cv.drawContours(img, [contour], -1, (0, 255, 0), 3)

    names = ["Hip", "Shoulder"]
    for name in names:
        left_bdr_p, right_bdr_p = find_symmetric_keypoints_on_boundary(contour, keypoints, name)
        left_p  = keypoints[POSE_BODY_25_BODY_PART_IDXS["".join(("L", name))]][:2]
        right_p = keypoints[POSE_BODY_25_BODY_PART_IDXS["".join(("R", name))]][:2]
        #cv.line(img, tuple(left_bdr_p.flatten()), tuple( right_bdr_p.flatten()), (0,0,255), thickness=10)
        #cv.drawMarker(img, tuple(left_bdr_p.flatten()), (255,0,0), markerSize=20, thickness=5)
        #cv.drawMarker(img, tuple(right_bdr_p.flatten()), (255,0,0), markerSize=20, thickness=5)
        #cv.drawMarker(img, tuple(left_p.flatten()), (0, 0, 255), markerSize=20, thickness=2)
        #cv.drawMarker(img, tuple(right_p.flatten()), (0, 0, 255), markerSize=20, thickness=2)

    for name in names:
        lname = "".join(('L', name))
        rname = "".join(('R', name))
        lpoint  = keypoints[POSE_BODY_25_BODY_PART_IDXS[lname]][:2]
        rpoint  = keypoints[POSE_BODY_25_BODY_PART_IDXS[rname]][:2]
        dir = lpoint - rpoint
        lpoint_ext = lpoint + 0.5 * dir
        rpoint_ext = rpoint - 0.5 * dir
        ipoints = isect_segment_contour(contour, lpoint_ext, rpoint_ext)
        for p in ipoints:
           cv.drawMarker(img, (int(p[0]), int(p[1])), (255, 0, 0), markerSize=20, thickness=5)
        if len(ipoints) == 2:
            cv.line(img, (int(ipoints[0][0]), int(ipoints[0][1])), (int(ipoints[1][0]), int(ipoints[1][1])), (255,255,255), thickness=10)

    lAnkle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rAnkle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    lAnkle, rAnkle = extend_segment(lAnkle, rAnkle, 3)
    cv.line(img, (int(lAnkle[0]), int(lAnkle[1])), (int(rAnkle[0]), int(rAnkle[1])), (0, 0, 255), thickness=10)

    seg = measure_acromial_height(contour, keypoints, left=True)
    cv.line(img, (int(seg[0][0]), int(seg[0][1])), (int(seg[1][0]), int(seg[1][1])), (255, 255, 255), thickness=10)
    seg = measure_acromial_height(contour, keypoints, left=False)
    cv.line(img, (int(seg[0][0]), int(seg[0][1])), (int(seg[1][0]), int(seg[1][1])), (255, 255, 255), thickness=10)

    return img

def load_silhouette(path, img):
    sil = cv.imread(path, cv.IMREAD_GRAYSCALE)
    sil = cv.resize(sil, (img.shape[1], img.shape[0]), cv.INTER_NEAREST)
    ret, sil = cv.threshold(sil, 200, maxval=255, type=cv.THRESH_BINARY)
    return sil

def int_tuple(vals):
    return tuple(int(v) for v in vals.flatten())

if __name__ == '__main__':
    OPENPOSE_MODEL_PATH = 'D:\Projects\Oh\\body_measure\openpose\models\\'

    ROOT_DIR    = 'D:\Projects\Oh\data\images\mobile\\'
    IMG_DIR     = f'{ROOT_DIR}pose\\'
    FRONT_IMG   = 'IMG_0917.JPG'
    SIDE_IMG    = 'IMG_0929.JPG'
    SILHOUETTE_DIR = f'{ROOT_DIR}graph_cut_result\\silhouette\\'

    OUT_DIR = f'{ROOT_DIR}pose_result\\'
    OUT_DIR_ALPHA_MAP = f'{ROOT_DIR}tri_map\\'
    OUT_MEASUREMENT_DIR = f'{ROOT_DIR}measurements\\'

    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = OPENPOSE_MODEL_PATH
    # Construct OpenPose object allocates GPU memory
    openpose = OpenPose(params)

    bone_widths = generate_bone_width()
    #n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU)/2)

    threshold = 0.01

    img_front = cv.imread(f'{IMG_DIR}{FRONT_IMG}')
    keypoints_front, img_front_pose = openpose.forward(img_front , True)
    sil_front = load_silhouette(f'{SILHOUETTE_DIR}{FRONT_IMG}', img_front)
    contour_front = find_largest_contour(sil_front)

    img_side = cv.imread(f'{IMG_DIR}{SIDE_IMG}')
    keypoints_side, img_side_pose= openpose.forward(img_side, True)
    sil_side = load_silhouette(f'{SILHOUETTE_DIR}{SIDE_IMG}', img_side)
    contour_side = find_largest_contour(sil_side)

    img_front = img_front_pose
    img_side = img_side_pose
    cv.drawContours(img_front, [contour_front], -1, (255, 255, 0), thickness=5)
    cv.drawContours(img_side,  [contour_side], -1, (255, 255, 0), thickness=5)

    # hip
    front_hip_bdr = estimate_front_hip_bdr_points(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_hip_bdr[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_hip_bdr[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_hip_bdr[0]), int_tuple(front_hip_bdr[1]), (0, 255, 255), thickness=5)

    side_hip_bdr = estimate_side_hip_bdr_points(contour_side, keypoints_side[0,:,:])
    cv.drawMarker(img_side, int_tuple(side_hip_bdr[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_side, int_tuple(side_hip_bdr[1]), (255, 0, 0), thickness=10)
    cv.line(img_side, int_tuple(side_hip_bdr[0]), int_tuple(side_hip_bdr[1]), (0, 255, 255), thickness=5)

    # waist
    front_waist_bdr = estimate_front_waist_bdr_points(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_waist_bdr[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_waist_bdr[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_waist_bdr[0]), int_tuple(front_waist_bdr[1]), (0, 255, 255), thickness=5)

    side_waist_bdr = estimate_side_waist_bdr_points(contour_side, keypoints_side[0,:,:])
    cv.drawMarker(img_side, int_tuple(side_waist_bdr[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_side, int_tuple(side_waist_bdr[1]), (255, 0, 0), thickness=10)
    cv.line(img_side, int_tuple(side_waist_bdr[0]), int_tuple(side_waist_bdr[1]), (0, 255, 255), thickness=5)

    # neck
    front_neck_bdr = estimate_front_neck_brd_points(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_neck_bdr[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_neck_bdr[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_neck_bdr[0]), int_tuple(front_neck_bdr[1]), (0, 255, 255), thickness=5)

    side_neck_bdr = estimate_front_neck_brd_points(contour_side, keypoints_side[0,:,:])
    cv.drawMarker(img_side, int_tuple(side_neck_bdr[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_side, int_tuple(side_neck_bdr[1]), (255, 0, 0), thickness=10)
    cv.line(img_side, int_tuple(side_neck_bdr[0]), int_tuple(side_neck_bdr[1]), (0, 255, 255), thickness=5)

    #inside leg
    front_points = estimate_front_inside_leg_bdr_points(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=5)

    side_points = estimate_side_inside_leg_bdr_points(contour_side, keypoints_side[0,:,:])
    cv.drawMarker(img_side, int_tuple(side_points[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_side, int_tuple(side_points[1]), (255, 0, 0), thickness=10)
    cv.line(img_side, int_tuple(side_points[0]), int_tuple(side_points[1]), (0, 255, 255), thickness=5)

    #shoulder
    front_points = estimate_front_shoulder_points(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=5)

    #arm
    front_points = estimate_front_arm(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=5)

    side_points = estimate_side_arm(contour_side, keypoints_side[0,:,:])
    cv.drawMarker(img_side, int_tuple(side_points[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_side, int_tuple(side_points[1]), (255, 0, 0), thickness=10)
    cv.line(img_side, int_tuple(side_points[0]), int_tuple(side_points[1]), (0, 255, 255), thickness=5)

    #around arm
    front_points = estimate_front_around_arm(contour_front, keypoints_front[0,:,:])
    cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
    cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
    cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=5)


    plt.subplot(121), plt.imshow(img_front[:,:,::-1])
    plt.subplot(122), plt.imshow(img_side[:,:,::-1])
    #plt.subplot(223), plt.imshow(img_front_pose[:,:,::-1])
    #plt.subplot(224), plt.imshow(img_side_pose[:,:,::-1])
    plt.show()
