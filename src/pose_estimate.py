import sys
import cv2 as cv
import os
from sys import platform
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from pathlib import Path
from openpose_util import (is_valid_keypoint, pair_length, pair_dir)
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points

OPENPOSE_PATH = 'D:\Projects\Oh\\body_measure\openpose\\build\python\openpose'
sys.path.append(OPENPOSE_PATH)
try:
    from openpose import *
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

POSE_BODY_25_PAIRS_RENDER_GPU = (
    1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16,
    16,
    18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24)
POSE_BODY_25_BODY_PARTS = \
    {
        0: "Nose",
        1: "Neck",
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

POSE_BODY_25_BODY_PART_IDXS = {v: k for k, v in POSE_BODY_25_BODY_PARTS.items()}

KEYPOINT_THRESHOLD = 0.01


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


def normalize(vec):
    len = linalg.norm(vec)
    if np.isclose(len, 0.0):
        return vec
    else:
        return (1.0 / len) * vec


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
    neck_nose = 50

    min_width = 5

    pairs = POSE_BODY_25_PAIRS_RENDER_GPU

    n_pairs = int(len(pairs) / 2)
    widths = np.zeros(n_pairs, dtype=np.int32)

    for i_pair in range(n_pairs):
        pair = (pairs[i_pair * 2], pairs[i_pair * 2 + 1])

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

def generate_fg_mask(img, keypoints, bone_widths):
    n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU) / 2)
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
    p0_ = (p0 + percent * dir).astype(np.int32)
    p1_ = (p1 - percent * dir).astype(np.int32)
    len0 = linalg.norm(dir)
    len1 = linalg.norm(p0_ - p1_)
    assert (len1 > len0)
    return p0_, p1_


def generate_bg_mask(img, keypoints, bone_widths):
    n_pairs = int(len(POSE_BODY_25_PAIRS_RENDER_GPU) / 2)

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

    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (50, 50)), iterations=4)

    return 255 - bg_mask


def gen_fg_bg_masks(img, keypoints, bone_widths):
    if keypoints.shape[0] < 1:
        fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        fg_mask = generate_fg_mask(img, keypoints, bone_widths)
        bg_mask = generate_bg_mask(img, keypoints, bone_widths)

        amap = np.zeros(fg_mask.shape, dtype=np.uint8)
        amap[fg_mask > 0] = 255
        amap[np.bitwise_and(np.bitwise_not(bg_mask > 0), np.bitwise_not(fg_mask > 0))] = 155

    # cv.imwrite(f'{OUT_DIR}{Path(img_path).name}', output_image)
    # cv.imwrite(f'{OUT_DIR_ALPHA_MAP}{Path(img_path).name}', amap)
    return fg_mask, bg_mask


def find_largest_contour(img_bi, app_type=cv.CHAIN_APPROX_TC89_L1):
    cnt, contours, _ = cv.findContours(img_bi, cv.RETR_LIST, app_type)
    largest_cnt = 0
    largest_area = -1
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt
    return largest_cnt.copy()


def closest_point_segment(p0, p1, p):
    p10 = p1 - p0
    a = np.dot(p, p10) - np.dot(p0, p10)
    b = np.dot(p10, p10)
    t = a / b
    if t > 1.0:
        return p1
    elif t < -1.0:
        return p0
    else:
        return p0 + t * p10


def dst_point_segment(p0, p1, p):
    tmp = closest_point_segment(p0, p1, p)
    tmp = p - tmp
    return np.norm(tmp)

def signed_dst_point_line(p0, dir, p):
    ortho = np.array([dir[1], -dir[0]])
    dst_0 = np.dot(p0, ortho)
    dst_1 = np.dot(p, ortho)
    return dst_1 - dst_0

def closest_point_contour(contour, point):
    diffs = contour - point
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    return np.argmin(dists)


def closest_dst_point_contour(contour, point):
    cls_idx = closest_point_contour(contour, point)
    return np.linalg.norm(contour[cls_idx, 0, :] - point)


def radius_search_on_contour(contour, point, radius):
    diffs = contour - point
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    mask = dists <= radius
    return np.where(mask == True)[0], dists[mask]

import sys

def closest_point_contour_segments(contour, point):
    contour = contour.astype(np.float32)
    point = point.astype(np.float32)
    cls_dst = sys.float_info.max
    cls_p = None
    n_point = len(contour)
    for i in range(n_point):
        p0 = contour[i, :, :].flatten()
        p1 = contour[(i + 1) % n_point, :, :].flatten()
        tmp_p = closest_point_segment(p0, p1, point)
        dst = np.linalg.norm(tmp_p - point)
        if dst < cls_dst:
            cls_dst = dst
            cls_p = tmp_p

    return cls_p

def extract_contour_chains(contour, point_checker):
    chains = []
    N = contour.shape[0]
    chain = []
    for i in range(N):
        if point_checker(contour[i, 0, :]):
            chain.append(contour[i, 0, :])
        else:
            if len(chain) > 0:
                chains.append(chain)
            chain = []
    return chains

def closest_point_points(point, points):
    g0 = Point(point)
    g1 = MultiPoint(points)
    closest = nearest_points(g0, g1)[1]
    return np.array([closest.x, closest.y])

def k_closest_point_points(point, points, k = 2):
    dsts = linalg.norm(points - point, axis=1)
    sorted_idxs = np.argsort(dsts)
    return points[sorted_idxs[:k]]

def isect_segment_contour(contour, p0, p1):
    a = LineString([p0, p1])
    cnt_points = [contour[i].flatten() for i in range(len(contour))]
    b = LineString(cnt_points)
    ipoints = a.intersection(b)
    return [(p.x, p.y) for p in ipoints]

def find_symmetric_keypoints_on_boundary(contour, keypoints, keypoint_name):
    left_name = "".join(("L", keypoint_name))
    right_name = "".join(("R", keypoint_name))
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

def measure_acromial_height(contour, keypoints, left=True):
    if left:
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    else:
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

    lAnkle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rAnkle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    lAnkle_ext, rAnkle_ext = extend_segment(lAnkle, rAnkle, 3)

    on_base_line = nearest_points(LineString([lAnkle_ext, rAnkle_ext]), Point(shoulder))[0]
    on_shoulder = nearest_points(LineString([contour[i].flatten() for i in range(len(contour))]), Point(shoulder))[0]

    return [(on_base_line.x, on_base_line.y), (on_shoulder.x, on_shoulder.y)]

def axis_front_view(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    axis = neck - midhip
    axis = normalize(axis)
    return axis


def axis_side_view(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
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
    return np.linalg.norm(neck - hip)


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
    s0 = midhip_side + axis_ortho * 0.5 * ref_length
    s1 = midhip_side - axis_ortho * 0.5 * ref_length
    ipoints = isect_segment_contour(contour_side, s0, s1)
    hip_bdr_side_0 = closest_point_points(s0, ipoints)
    hip_bdr_side_1 = closest_point_points(s1, ipoints)
    return np.vstack([hip_bdr_side_0, hip_bdr_side_1])


def estimate_front_waist_bdr_points(contour, keypoints):
    axis_ortho = orthor_dir(axis_front_view(keypoints))

    lshouder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    lelbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    len = linalg.norm(lshouder - lelbow)

    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]

    dir = normalize(midhip - neck)
    waist_center = neck + len * dir

    contour_string = LineString([contour[i, 0, :] for i in range(contour.shape[0])])

    end_point_0 = waist_center + len * axis_ortho
    end_point_1 = waist_center - len * axis_ortho
    horizon_line = LineString([end_point_0, end_point_1])

    isect_points = horizon_line.intersection(contour_string)

    p0 = closest_point_points(waist_center + 0.1 * len * axis_ortho, isect_points)
    p1 = closest_point_points(waist_center - 0.1 * len * axis_ortho, isect_points)

    return np.vstack([p0, p1])


def estimate_side_waist_bdr_points(contour, keypoints):
    axis = axis_side_view(keypoints)
    axis_ortho = orthor_dir(axis)

    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    dir = neck - midhip
    waist_center = midhip + 0.333 * dir

    ref_length = neck_hip_length(keypoints)
    s0 = waist_center + axis_ortho * 0.5 * ref_length
    s1 = waist_center - axis_ortho * 0.5 * ref_length

    ipoints = isect_segment_contour(contour, s0, s1)

    waist_bdr_side_0 = closest_point_points(s0, ipoints)
    waist_bdr_side_1 = closest_point_points(s1, ipoints)

    return np.vstack([waist_bdr_side_0, waist_bdr_side_1])

def estimate_front_neck_brd_points(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    mid = 0.5 * (neck + nose)
    radius = 0.5 * linalg.norm(neck - nose)

    def point_checker(p):
        if p[1] > nose[1] and p[1] < neck[1]:
            return True
        else:
            return False

    chains = extract_contour_chains(contour, point_checker)

    chain_0 = LineString(chains[0])
    chain_1 = LineString(chains[1])

    p0, p1 = nearest_points(chain_0, chain_1)

    return np.vstack([p0, p1])


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
    lankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    ankle = 0.5 * (lankle + rankle)
    return np.vstack([midhip, ankle])


def estimate_side_inside_leg_bdr_points(contour, keypoints):
    return estimate_front_inside_leg_bdr_points(contour, keypoints)

def estimate_front_shoulder_points(contour, curvatures, keypoints):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    lshoulder = Point(lshoulder)
    rshoulder = Point(rshoulder)

    N = len(contour)

    l_cls_dst = 9999; r_cls_dst = 9999
    l_cls = np.zeros(2); r_cls = np.zeros(2)

    for i in range(N):
        p = contour[i,0,:]
        p_n = contour[(i+1)%N,0,:]
        segment = LineString([p, p_n])
        #left shoulder
        if p[0] > lshoulder.x or p_n[0] > lshoulder.x:
            cls_pnt = nearest_points(segment, lshoulder)[0]
            dst = cls_pnt.distance(lshoulder)
            if dst < l_cls_dst:
                l_cls_dst = dst
                l_cls = cls_pnt

        #right shoulder
        if p[0] > rshoulder.x or p_n[0] > rshoulder.x:
            cls_pnt = nearest_points(segment, rshoulder)[0]
            dst = cls_pnt.distance(rshoulder)
            if dst < r_cls_dst:
                r_cls_dst = dst
                r_cls = cls_pnt
    return np.vstack([l_cls, r_cls])

def estimate_front_armpit(contour, curvatures, keypoints):
    lelbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    relbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]

    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]

    lhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]

    radius = np.linalg.norm(lelbow - lshoulder)

    lelbow_shoulder = pair_dir(keypoints, 'LElbow', 'LShoulder')
    lshoulder_neck  = pair_dir(keypoints, 'LShoulder', 'Neck')
    lpoint_idxs, _ = radius_search_on_contour(contour, lshoulder, radius)
    larmpit_idx = -1
    nearst_dst  = 999999999
    for idx in lpoint_idxs:
        # armpit landmark must be somewhere beween hip and shoulder
        if contour[idx, 0, 1] > lshoulder[1] and contour[idx, 0, 1] < lhip[1]:
            cur_point = contour[idx, 0, :]
            sign_dst_0 = signed_dst_point_line(lelbow, lelbow_shoulder, cur_point)
            sign_dst_1 = signed_dst_point_line(lshoulder, lshoulder_neck, cur_point)
            if (sign_dst_0 * sign_dst_1 ) > 0:
                if np.abs(sign_dst_1) < nearst_dst:
                    nearst_dst = np.abs(sign_dst_1)
                    larmpit_idx = idx

    relbow_shoulder = pair_dir(keypoints, 'RElbow', 'RShoulder')
    rshoulder_neck  = pair_dir(keypoints, 'RShoulder', 'Neck')
    rpoint_idxs, _ = radius_search_on_contour(contour, rshoulder, radius)
    rarmpit_idx = -1
    nearst_dst  = 999999999
    for idx in rpoint_idxs:
        # armpit landmark must be somewhere beween hip and shoulder
        if contour[idx, 0, 1] > rshoulder[1] and contour[idx, 0, 1] < rhip[1]:
            cur_point = contour[idx, 0, :]
            sign_dst_0 = signed_dst_point_line(relbow, relbow_shoulder, cur_point)
            sign_dst_1 = signed_dst_point_line(rshoulder, rshoulder_neck, cur_point)
            if (sign_dst_0 * sign_dst_1 ) > 0:
                if np.abs(sign_dst_1) < nearst_dst:
                    nearst_dst = np.abs(sign_dst_1)
                    rarmpit_idx = idx

    return np.vstack([contour[larmpit_idx, 0, :], contour[rarmpit_idx, 0, :]])


def estimate_front_elbow(contour, keypoints, left = False):
    if left == True:
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    else:
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

    dir = elbow - shoulder
    len = linalg.norm(dir)
    dir_orth = orthor_dir(normalize(dir))

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    end_point_0 = elbow + 0.5 * len * dir_orth
    end_point_1 = elbow - 0.5 * len * dir_orth
    ortho_line = LineString([end_point_0, end_point_1])

    isect_points = ortho_line.intersection(contour_string)
    isect_points = np.array([(p.x, p.y) for p in isect_points])

    p0, p1 = k_closest_point_points(elbow, isect_points)

    return np.vstack([p0, p1])

#def estimate_side_arm(contour, keypoints):
#    return estimate_front_arm(contour, keypoints)

def estimate_front_wrist(contour, keypoints, left = False):
    if left == True:
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
        wrist = keypoints[POSE_BODY_25_BODY_PART_IDXS['LWrist']][:2]
    else:
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]
        wrist = keypoints[POSE_BODY_25_BODY_PART_IDXS['RWrist']][:2]

    dir = elbow - wrist
    len = linalg.norm(dir)
    dir_orth = orthor_dir(normalize(dir))

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    end_point_0 = wrist + 0.3 * len * dir_orth
    end_point_1 = wrist - 0.3 * len * dir_orth
    ortho_line = LineString([end_point_0, end_point_1])

    isect_points = ortho_line.intersection(contour_string)
    isect_points = np.array([(p.x, p.y) for p in isect_points])

    p0, p1 = k_closest_point_points(wrist, isect_points)

    return np.vstack([p0, p1])

def load_silhouette(path, img):
    sil = cv.imread(path, cv.IMREAD_GRAYSCALE)
    sil = cv.resize(sil, (img.shape[1], img.shape[0]), cv.INTER_NEAREST)
    ret, sil = cv.threshold(sil, 200, maxval=255, type=cv.THRESH_BINARY)
    return sil

def fix_silhouette(sil):
    sil = cv.morphologyEx(sil, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, ksize=(3,3)))
    return sil

def load_body_template_contour(path):
    gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
    bi = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    contour = find_largest_contour(bi)
    return contour

def int_tuple(vals):
    return tuple(int(v) for v in vals.flatten())

from scipy.ndimage import filters

def smooth_contour(contour, sigma=3):
    contour_new = contour.astype(np.float32)
    contour_new[:, 0, 0] = filters.gaussian_filter1d(contour[:, 0, 0], sigma=sigma)
    contour_new[:, 0, 1] = filters.gaussian_filter1d(contour[:, 0, 1], sigma=sigma)
    return contour_new.astype(np.int32)

from scipy import interpolate
def calc_curvature_polyfit(img, contour, win_size, N_resample=1000):
    half_win_size = int(0.5 * win_size)
    N = N_resample

    x = contour[:, 0, 0].astype(np.float64)
    y = contour[:, 0, 1].astype(np.float64)
    tck, u = interpolate.splprep([x, y], s=1)
    xnew = np.arange(0, 1, 1.0 / float(N))
    ynew = interpolate.splev(xnew, tck, der=0)
    contour_new = np.zeros((N, 1, 2), dtype=np.int32)
    contour_new[:, 0, 0] = ynew[0].astype(np.int32)
    contour_new[:, 0, 1] = ynew[1].astype(np.int32)

    curvatures = np.zeros(N, np.float32)
    for i in range(half_win_size, N - half_win_size):
        local_X = contour_new[i - half_win_size: i + half_win_size, 0, 0]
        local_Y = contour_new[i - half_win_size: i + half_win_size, 0, 1]
        coeffs = np.polyfit(local_X, local_Y, 2)
        curvatures[i] = coeffs[0]

    print(f'curvature min = {curvatures.min()} , max = {curvatures.max()}')
    return contour_new, curvatures


def calc_curvature_dot_product(img, contour):
    N = contour.shape[0]
    curvatures = np.zeros(N, np.float32)
    for i in range(N):
        iprev = i - 1 if i > 0 else N - 1
        inext = (i + 1) % N
        prev = contour[iprev, 0, :]
        next = contour[inext, 0, :]
        p = contour[i, 0, :]
        prev_dir = normalize(p - prev)
        next_dir = normalize(next - p)
        curvatures[i] = 1 - np.abs(np.dot(prev_dir, next_dir))
    print(f'curvature min = {curvatures.min()} , max = {curvatures.max()}')
    return contour, curvatures


def display_contour_on_img(img, contour, curvatures=None, out_fig_path=None):
    # curvatures = (curvatures - curvatures.min())/(curvatures.max() - curvatures.min())
    if curvatures != None:
        cm = plt.get_cmap('jet')
        colors = (cm(curvatures) * 255).astype(np.int32)
        for i in range(len(curvatures)):
            pos = (contour[i, 0, 0], contour[i, 0, 1])
            cv.drawMarker(img, pos, (int(colors[2]), int(colors[1]), int(colors[0])), markerType=cv.MARKER_CROSS,
                          markerSize=10, thickness=10)

    cv.drawContours(img, [contour], -1, (255, 255, 255), thickness=2)
    plt.imshow(img[:, :, ::-1])
    if out_fig_path != None:
        plt.savefig(out_fig_path, dpi=1000)
    plt.show()


if __name__ == '__main__':
    OPENPOSE_MODEL_PATH = 'D:\Projects\Oh\\body_measure\openpose\models\\'

    ROOT_DIR = 'D:\Projects\Oh\data\images\mobile\oh_images\\'
    IMG_DIR = f'{ROOT_DIR}images\\'
    SILHOUETTE_DIR = f'{ROOT_DIR}\silhouette_post\\'
    #OUT_DIR = f'{ROOT_DIR}pose_result\\'
    OUT_MEASUREMENT_DIR = f'{ROOT_DIR}measurements_deeplab_postprocess\\'

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

    MARKER_SIZE = 5
    MARKER_THICKNESS = 5
    LINE_THICKNESS = 2

    for img_path in Path(IMG_DIR).glob('*.*'):

        img_front = cv.imread(str(img_path))
        keypoints_front, img_front_pose = openpose.forward(img_front, True)
        sil_front = load_silhouette(f'{SILHOUETTE_DIR}{img_path.name}', img_front)
        sil_front = fix_silhouette(sil_front)
        contour_front = find_largest_contour(sil_front, cv.CHAIN_APPROX_NONE)
        contour_front = smooth_contour(contour_front, 3)
        contour_front = cv.approxPolyDP(contour_front, 3, closed=True)
        contour_front, curvatures_front = calc_curvature_dot_product(sil_front, contour_front)

        # display_contour_on_img(img_front_pose, contour_new, curvatures = curvatures)

        # img_side = cv.imread(f'{IMG_DIR}{SIDE_IMG}')
        # keypoints_side, img_side_pose = openpose.forward(img_side, True)
        # sil_side = load_silhouette(f'{SILHOUETTE_DIR}{SIDE_IMG}', img_side)
        # contour_side = find_largest_contour(sil_side)
        # contour_side = smooth_contour(contour_side, 3)
        # contour_side = cv.approxPolyDP(contour_side, 7, closed=True)
        # contour_side, curvatures_side = calc_curvature_dot_product(sil_side, contour_side)

        img_front = img_front_pose
        cv.drawContours(img_front, [contour_front], -1, (255, 255, 0), thickness=3)
        cm = plt.get_cmap('jet')
        colors = (cm(curvatures_front) * 255).astype(np.int32)
        for i in range(len(curvatures_front)):
            pos = (contour_front[i, 0, 0], contour_front[i, 0, 1])
            color = colors[i]
            cv.drawMarker(img_front, pos, (int(color[2]), int(color[1]), int(color[0])), markerType=cv.MARKER_CROSS,
                          markerSize=MARKER_SIZE, thickness=MARKER_THICKNESS)

        # img_side = img_side_pose
        # cv.drawContours(img_side, [contour_side], -1, (255, 255, 0), thickness=3)
        # colors = (cm(curvatures_side) * 255).astype(np.int32)
        # for i in range(len(curvatures_side)):
        #     pos = (contour_side[i, 0, 0], contour_side[i, 0, 1])
        #     color = colors[i]
        #     cv.drawMarker(img_side, pos, (int(color[2]), int(color[1]), int(color[0])), markerType=cv.MARKER_CROSS,
        #                   markerSize=15, thickness=15)

        # shoulder
        front_points = estimate_front_shoulder_points(contour_front, curvatures_front, keypoints_front[0, :, :])
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # neck
        front_points = estimate_front_neck_brd_points(contour_front, keypoints_front[0, :, :])
        # cv.drawMarker(img_front, int_tuple(front_neck_bdr[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_neck_bdr[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # side_neck_bdr = estimate_side_neck_brd_points(contour_side, keypoints_side[0, :, :])
        # # cv.drawMarker(img_side, int_tuple(side_neck_bdr[0]), (255, 0, 0), thickness=10)
        # # cv.drawMarker(img_side, int_tuple(side_neck_bdr[1]), (255, 0, 0), thickness=10)
        # cv.line(img_side, int_tuple(side_neck_bdr[0]), int_tuple(side_neck_bdr[1]), (0, 255, 255), thickness=3)

        # armpit
        front_points = estimate_front_armpit(contour_front, curvatures_front, keypoints_front[0, :, :])
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # left elbow
        front_points = estimate_front_elbow(contour_front, keypoints_front[0, :, :], left=True)
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # front elbow
        front_points = estimate_front_elbow(contour_front, keypoints_front[0, :, :], left=False)
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # arm
        front_points = estimate_front_wrist(contour_front, keypoints_front[0, :, :], left=True)
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        front_points = estimate_front_wrist(contour_front, keypoints_front[0, :, :], left=False)
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # hip
        front_hip_bdr = estimate_front_hip_bdr_points(contour_front, keypoints_front[0, :, :])
        # cv.drawMarker(img_front, int_tuple(front_hip_bdr[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_hip_bdr[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_hip_bdr[0]), int_tuple(front_hip_bdr[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # side_hip_bdr = estimate_side_hip_bdr_points(contour_side, keypoints_side[0, :, :])
        # # cv.drawMarker(img_side, int_tuple(side_hip_bdr[0]), (255, 0, 0), thickness=10)
        # # cv.drawMarker(img_side, int_tuple(side_hip_bdr[1]), (255, 0, 0), thickness=10)
        # cv.line(img_side, int_tuple(side_hip_bdr[0]), int_tuple(side_hip_bdr[1]), (0, 255, 255), thickness=5)

        # waist
        front_waist_bdr = estimate_front_waist_bdr_points(contour_front, keypoints_front[0, :, :])
        # cv.drawMarker(img_front, int_tuple(front_waist_bdr[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_waist_bdr[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_waist_bdr[0]), int_tuple(front_waist_bdr[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # side_waist_bdr = estimate_side_waist_bdr_points(contour_side, keypoints_side[0, :, :])
        # # cv.drawMarker(img_side, int_tuple(side_waist_bdr[0]), (255, 0, 0), thickness=10)
        # # cv.drawMarker(img_side, int_tuple(side_waist_bdr[1]), (255, 0, 0), thickness=10)
        # cv.line(img_side, int_tuple(side_waist_bdr[0]), int_tuple(side_waist_bdr[1]), (0, 255, 255), thickness=3)

        # inside leg
        front_points = estimate_front_inside_leg_bdr_points(contour_front, keypoints_front[0, :, :])
        # cv.drawMarker(img_front, int_tuple(front_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_front, int_tuple(front_points[1]), (255, 0, 0), thickness=10)
        cv.line(img_front, int_tuple(front_points[0]), int_tuple(front_points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        # side_points = estimate_side_inside_leg_bdr_points(contour_side, keypoints_side[0, :, :])
        # # cv.drawMarker(img_side, int_tuple(side_points[0]), (255, 0, 0), thickness=10)
        # # cv.drawMarker(img_side, int_tuple(side_points[1]), (255, 0, 0), thickness=10)
        # cv.line(img_side, int_tuple(side_points[0]), int_tuple(side_points[1]), (0, 255, 255), thickness=3)

        #side_points = estimate_side_arm(contour_side, keypoints_side[0, :, :])
        # cv.drawMarker(img_side, int_tuple(side_points[0]), (255, 0, 0), thickness=10)
        # cv.drawMarker(img_side, int_tuple(side_points[1]), (255, 0, 0), thickness=10)
        #cv.line(img_side, int_tuple(side_points[0]), int_tuple(side_points[1]), (0, 255, 255), thickness=5)

        plt.subplot(111), plt.imshow(img_front[:, :, ::-1])
        #plt.subplot(122), plt.imshow(img_side[:, :, ::-1])
        #plt.show()
        plt.savefig(f'{OUT_MEASUREMENT_DIR}{img_path.name}', dpi=1000)
