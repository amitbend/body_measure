import sys
import cv2 as cv
import os
from sys import platform
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from pathlib import Path
import openpose_util as ut
from openpose_util import (is_valid_keypoint, is_valid_keypoint_1, pair_length, pair_dir, find_pose, int_tuple)
from pose_to_trimap import gen_fg_bg_masks, head_center_estimate
from silhouette_refine import refine_silhouette_side_img, refine_silhouette_front_img
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points

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

def extend_segment(p0, p1, percent):
    dir = p0 - p1
    p0_ = (p0 + percent * dir).astype(np.int32)
    p1_ = (p1 - percent * dir).astype(np.int32)
    len0 = linalg.norm(dir)
    len1 = linalg.norm(p0_ - p1_)
    assert (len1 > len0)
    return p0_, p1_


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
    if len(points) > 1:
        g1 = MultiPoint(points)
        closest = nearest_points(g0, g1)[1]
        return np.array([closest.x, closest.y])
    else:
        return np.array([g0.x, g0.y])

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

def estimate_front_points(contour_front, keypoints_front):
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

def estimate_front_crotch(contour, keypoints):
    lhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    x_range = (min(lhip[0], rhip[0]), max(lhip[0], rhip[0]))
    min_y = 9999
    crotch = np.zeros(2)
    for i in range(contour.shape[0]):
        p = contour[i,0,:]
        if p[0] > x_range[0] and p[0] < x_range[1] and p[1] > lhip[1]:
            if p[1] <  min_y:
                min_y = p[1]
                crotch = p
    return crotch

def estimate_front_waist_points(contour, keypoints):
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

def estimate_front_neck_points(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]

    def point_checker(p):
        if p[1] > nose[1] and p[1] < neck[1]:
            return True
        else:
            return False

    chains = extract_contour_chains(contour, point_checker)

    chain_0 = LineString(chains[0])
    chain_1 = LineString(chains[1])

    p0, p1 = nearest_points(chain_0, chain_1)

    #there're often noise around neck => choose the intersection point that give the shortest distance
    neck_nose = LineString([nose, neck])
    dst_0 = p0.distance(neck_nose)
    dst_1 = p1.distance(neck_nose)

    p0 = p0 if dst_0 < dst_1 else p1
    p_on_bone = nearest_points(p0, neck_nose)[1]

    p0 = np.array(p0.coords).flatten()
    p_on_bone = np.array(p_on_bone.coords).flatten()
    p1 = p_on_bone + (p_on_bone - p0)

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


def estimate_front_inside_leg_points(contour, keypoints):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    lankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    ankle = 0.5 * (lankle + rankle)
    return np.vstack([midhip, ankle])

def estimate_side_inside_leg_bdr_points(contour, keypoints):
    return estimate_front_inside_leg_points(contour, keypoints)

def estimate_front_knee_points(contour, keypoints, is_left):
    if is_left:
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
        ankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    else:
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
        ankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]

    dir = knee - ankle
    len = linalg.norm(dir)
    dir_orth = orthor_dir(normalize(dir))

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    end_point_0 = knee + 0.5 * len * dir_orth
    end_point_1 = knee - 0.5 * len * dir_orth
    ortho_line = LineString([end_point_0, end_point_1])

    isect_points = ortho_line.intersection(contour_string)
    if isect_points.geom_type == 'Point':
        print('estimate_front_knee_points: just found one intersection point')
        return np.vstack([knee, knee])
    elif isect_points.geom_type == 'MultiPoint':
        isect_points = np.array([(p.x, p.y) for p in isect_points])
        p0, p1 = k_closest_point_points(knee, isect_points)
        return np.vstack([p0, p1])
    else:
        print('estimate_front_knee_points: no intersection found!')
        return np.vstack([knee, knee + 0.3 *(knee-ankle)])

def estimate_front_thigh(contour, keypoints, is_left):
    if is_left:
        hip  =  keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
        knee =  keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
        dir = orthor_dir(knee - hip)
    else:
        hip  =  keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
        knee =  keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
        dir = orthor_dir(hip - knee)

    crotch = estimate_front_crotch(contour, keypoints)
    p0 = crotch + 0.05 * dir
    p1 = crotch + dir

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    isect_points = LineString([p0, p1]).intersection(contour_string)
    if isect_points.geom_type == 'Point':
        return np.vstack([crotch, (isect_points.x, isect_points.y)])
    elif isect_points.geom_type == 'MultiPoint':
        isect_points = np.array([(p.x, p.y) for p in isect_points])
        p0, p1 = k_closest_point_points(p0, isect_points)
        return np.vstack([p0, p1])
    else:
        print('estimate_front_thigh: no intersection found!')
        return np.vstack([crotch, crotch+ 0.3 * dir])

def estimate_front_shoulder_points(contour, keypoints):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    length = linalg.norm(rshoulder- lshoulder)

    contour_string = LineString(contour[i].flatten() for i in range(len(contour)))

    up_dir = np.array([0, -1.0])
    p0 = lshoulder
    p1 = p0 + 0.5 * length * (normalize(lshoulder - rshoulder) + up_dir)
    isect_points = LineString([p0, p1]).intersection(contour_string)
    shoulder_contour_0 = nearest_points(Point(p0), isect_points)[1]

    p0 = rshoulder
    p1 = p0 + 0.5 * length * (normalize(rshoulder - lshoulder) + up_dir)
    isect_points = LineString([p0, p1]).intersection(contour_string)
    shoulder_contour_1 =  nearest_points(Point(p0), isect_points)[1]

    return np.vstack([(shoulder_contour_0.x, shoulder_contour_0.y), (shoulder_contour_1.x, shoulder_contour_1.y)])

def estimate_front_armpit(contour, keypoints):
    lelbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    relbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]

    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

    lhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]

    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]

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

    #take the highest armpit
    larmpit = contour[larmpit_idx, 0, :]
    rarmpit = contour[rarmpit_idx, 0, :]
    if larmpit[1] < rarmpit[1]:
        highest_armpit = larmpit
        half_length = Point(highest_armpit).distance(LineString([neck, midhip]))
        mirror_armpit = highest_armpit + 2 * half_length*normalize(rshoulder - lshoulder)
    else:
        highest_armpit = rarmpit
        half_length = Point(highest_armpit).distance(LineString([neck, midhip]))
        mirror_armpit = highest_armpit + 2 * half_length*normalize(lshoulder - rshoulder)

    return np.vstack([highest_armpit, mirror_armpit])

def estimate_side_hip(contour, keypoints):
    if is_valid_keypoint_1(keypoints, 'LHip') and is_valid_keypoint_1(keypoints, 'LKnee'):
        hip  = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
        knee   = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
        dir = orthor_dir(knee - hip)
    elif is_valid_keypoint_1(keypoints, 'RHip') and is_valid_keypoint_1(keypoints, 'RKnee'):
        hip  = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
        knee   = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
        dir = orthor_dir(knee - hip)
    elif is_valid_keypoint_1(keypoints, 'MidHip') and is_valid_keypoint_1(keypoints, 'Neck'):
        hip   = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
        neck  = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
        dir = orthor_dir(neck - hip)
    else:
        dir = np.array([500,0])

    p0 = hip + 0.7 * dir
    p1 = hip - 0.7 * dir

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    isect_points = LineString([p0, p1]).intersection(contour_string)
    if isect_points.geom_type == 'MultiPoint':
        points = np.array([(p.x, p.y) for p in isect_points])
        p0, p1 = k_closest_point_points(hip, points , k=2)
        return np.vstack([p0, p1])
    else:
        print('estimate_side_hip: up to two intersection point needed!')
        return np.vstack([p0, p1])

def estimate_side_waist(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    midhip_neck = midhip - neck
    if is_valid_keypoint_1(keypoints, 'LElbow'):
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    elif is_valid_keypoint_1(keypoints, 'RElbow'):
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]
    else:
        print('estimate_side_waist: missing LElbow or RElbow keypoints in side image!')
        elbow = neck + 0.6 * midhip_neck

    if is_valid_keypoint_1(keypoints, 'LShoulder'):
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    elif is_valid_keypoint_1(keypoints, 'RShoulder'):
        shoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    else:
        print('estimate_side_waist: missing LShoulder or RShoulder keypoints in side image!')
        shoulder = neck

    waist =  neck + linalg.norm(shoulder-elbow)*normalize(midhip_neck)
    p0 = waist + orthor_dir(midhip_neck)
    p1 = waist - orthor_dir(midhip_neck)
    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    isect_points = LineString([p0, p1]).intersection(contour_string)
    if isect_points.geom_type == 'MultiPoint':
        points = np.array([(p.x, p.y) for p in isect_points])
        p0, p1 = k_closest_point_points(elbow, points , k=2)
        return np.vstack([p0, p1])
    else:
        print('estimate_side_waist: up to two intersection point needed!')
        return np.vstack([p0, p1])

def estimate_side_largest_waist(contour, keypoints):
    if is_valid_keypoint_1(keypoints, 'LElbow'):
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    elif is_valid_keypoint_1(keypoints, 'RElbow'):
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]
    else:
        print('estimate_side_largest_waist: missing LElbow or RElbow keypoints in side image!')
        return np.vstack([np.zeros(2), np.zeros(2)])

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    hor_seg = orthor_dir(neck-midhip)


    start_p = midhip + 0.25 * (elbow - midhip)
    end_p = midhip + 0.75 * (elbow - midhip)

    n_samples = 10
    largest_dst = 0
    waist_0 = None; waist_1 = None
    for i in range(n_samples):
        p = start_p + float(i)/float(n_samples) * (end_p - start_p)
        p0 = p + hor_seg
        p1 = p - hor_seg
        isect_points = LineString([p0, p1]).intersection(contour_string)
        if isect_points.geom_type == 'MultiPoint':
            points = np.array([(p.x, p.y) for p in isect_points])
            isct_p0, isct_p1 = k_closest_point_points(elbow, points, k=2)
            dst = linalg.norm(isct_p0 - isct_p1)
            if dst > largest_dst:
                largest_dst = dst
                waist_0, waist_1 = isct_p0, isct_p1

    if waist_0 is not None:
        return np.vstack([waist_0, waist_1])
    else:
        return np.vstack([np.zeros(2), np.zeros(2)])

def estimate_side_chest(contour, keypoints):
    if is_valid_keypoint_1(keypoints, 'LElbow'):
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    elif is_valid_keypoint_1(keypoints, 'RElbow'):
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]
    else:
        print('estimate_side_waist: missing LElbow or RElbow keypoints in side image!')
        return np.vstack([np.zeros(2), np.zeros(2)])

    contour_string = LineString([contour[i,0,:] for i in range(contour.shape[0])])
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    hor_seg = orthor_dir(neck-midhip)
    n_samples = 10
    largest_dst = 0
    chest_0 = None; chest_1 = None
    for i in range(n_samples):
        p = elbow + float(i)/float(n_samples) * (neck - elbow)
        p0 = p + hor_seg
        p1 = p - hor_seg
        isect_points = LineString([p0, p1]).intersection(contour_string)
        if isect_points.geom_type == 'MultiPoint':
            points = np.array([(p.x, p.y) for p in isect_points])
            isct_p0, isct_p1 = k_closest_point_points(elbow, points, k=2)
            dst = linalg.norm(isct_p0 - isct_p1)
            if dst > largest_dst:
                largest_dst = dst
                chest_0, chest_1 = isct_p0, isct_p1

    if chest_0 is not None:
        return np.vstack([chest_0, chest_1])
    else:
        return np.vstack([np.zeros(2), np.zeros(2)])

def estimate_side_height(contour, keypoints):
    lowest = 99999
    for i in range(contour.shape[0]):
        p = contour[i,0,:]
        if p[1] < lowest:
            lowest = p[1]
            head_tip = p

    if is_valid_keypoint_1(keypoints, 'LHeel') and is_valid_keypoint_1(keypoints, 'RHeel'):
        lankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHeel']][:2]
        rankle  = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHeel']][:2]
        bottom = 0.5 * (lankle + rankle)
        return np.vstack([head_tip, bottom])
    else:
        contour_string = LineString([contour[i, 0, :] for i in range(contour.shape[0])])
        p0 = head_tip
        p1 = head_tip + 9999*np.array([0,1])
        isect_points = LineString([p0, p1]).intersection(contour_string)
        lowest = 0
        for p in isect_points:
            if p.y > lowest:
                lowest = p.y
                bottom = (p.x, p.y)

        return np.vstack([head_tip, bottom])

def estimate_front_elbow(contour, keypoints, left = False):
    if left == True:
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
        wrist= keypoints[POSE_BODY_25_BODY_PART_IDXS['LWrist']][:2]
    else:
        elbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]
        wrist = keypoints[POSE_BODY_25_BODY_PART_IDXS['RWrist']][:2]

    dir = elbow - wrist
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
    contour = ut.find_largest_contour(bi)
    return contour

def estimate_front_measures(contour, keypoints):
    landmarks = {}

    # shoulder
    points = estimate_side_height(contour, keypoints[0, :, :])
    landmarks['Height'] = points

    # shoulder
    points = estimate_front_shoulder_points(contour, keypoints[0, :, :])
    landmarks['Shoulder'] = points

    # neck
    points = estimate_front_neck_points(contour, keypoints[0, :, :])
    landmarks['Neck'] = points

    # armpit
    points = estimate_front_armpit(contour, keypoints[0, :, :])
    landmarks['Armpit'] = points

    # left elbow
    points = estimate_front_elbow(contour, keypoints[0, :, :], left=True)
    landmarks['LElbow'] = points
    points = estimate_front_elbow(contour, keypoints[0, :, :], left=False)
    landmarks['RElbow'] = points

    #wrist
    points = estimate_front_wrist(contour, keypoints[0, :, :], left=True)
    landmarks['LWrist'] = points
    points = estimate_front_wrist(contour, keypoints[0, :, :], left=False)
    landmarks['RWrist'] = points

    # hip
    points = estimate_front_points(contour, keypoints[0, :, :])
    landmarks['Hip'] = points

    # waist
    points = estimate_front_waist_points(contour, keypoints[0, :, :])
    landmarks['Waist'] = points

    # inside leg
    points = estimate_front_inside_leg_points(contour, keypoints[0, :, :])
    landmarks['InsideLeg'] = points

    points = estimate_front_knee_points(contour, keypoints[0, :, :], is_left=True)
    landmarks['LKnee'] = points
    points = estimate_front_knee_points(contour, keypoints[0, :, :], is_left=False)
    landmarks['RKnee'] = points

    points = estimate_front_thigh(contour, keypoints[0, :, :], is_left=False)
    landmarks['LThigh'] = points
    points = estimate_front_thigh(contour, keypoints[0, :, :], is_left=True)
    landmarks['RThigh'] = points

    return landmarks

def estimate_side_measures(contour, keypoints_s):
    landmarks = {}

    points = estimate_side_hip(contour, keypoints_s[0,:,:])
    landmarks['Hip'] = points

    points = estimate_side_waist(contour, keypoints_s[0,:,:])
    landmarks['Waist'] = points

    #points = estimate_side_largest_waist(contour, keypoints_s[0,:,:])
    #landmarks['LargestWaist'] = points

    points = estimate_side_chest(contour, keypoints_s[0,:,:])
    landmarks['Chest'] = points

    points = estimate_side_height(contour, keypoints_s[0,:,:])
    landmarks['Height'] = points

    return landmarks

if __name__ == '__main__':
    ROOT_DIR = '/home/khanhhh/data_1/projects/Oh/data/oh_mobile_images/'
    IMG_DIR = f'{ROOT_DIR}images/'
    SILHOUETTE_DIR = f'{ROOT_DIR}silhouette_refined/'
    OUT_MEASUREMENT_DIR = f'{ROOT_DIR}measurements/'

    MARKER_SIZE = 5
    MARKER_THICKNESS = 5
    LINE_THICKNESS = 2

    for f in Path(OUT_MEASUREMENT_DIR).glob('*.*'):
        os.remove(f)

    front_img_info = {}
    for img_path in Path(IMG_DIR).glob('*.*'):
        if 'front_' not in str(img_path):
            continue
        print(img_path)

        img_org = cv.imread(str(img_path))
        keypoints, img_pose = find_pose(img_org)

        sil = load_silhouette(f'{SILHOUETTE_DIR}{img_path.name}', img_org)
        contour = ut.find_largest_contour(sil)
        cv.drawContours(img_pose, [contour], -1, color=(255,0,0), thickness=3)
        img = img_pose

        landmarks = estimate_front_measures(contour, keypoints)
        for name, points in landmarks.items():
            cv.line(img, int_tuple(points[0]), int_tuple(points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        front_img_info[img_path.name] = {'keypoints':keypoints, 'landmarks':landmarks}

        plt.imshow(img[:, :, ::-1])
        #plt.show()
        plt.savefig(f'{OUT_MEASUREMENT_DIR}{img_path.name}', dpi=1000)

    mapping = {}
    mapping['side_IMG_1935.JPG'] = 'front_IMG_1928.JPG'
    mapping['side_IMG_1941.JPG'] = 'front_IMG_1939.JPG'
    mapping['side_8E2593C4-35E4-4B49-9B89-545AC906235C.jpg'] = 'front_9EF020C7-2CC9-4171-8378-60132015289D.jpg'

    for img_path in Path(IMG_DIR).glob('*.*'):
        if 'side_' not in str(img_path):
            continue

        if img_path.name not in mapping:
            continue

        front_name = mapping[img_path.name]
        if front_name not in front_img_info:
            print('Error: no front image information found')
            continue

        print(img_path)

        img_org = cv.imread(str(img_path))
        keypoints_s, img_pose = find_pose(img_org)

        sil = load_silhouette(f'{SILHOUETTE_DIR}{img_path.name}', img_org)
        contour = ut.find_largest_contour(sil)
        cv.drawContours(img_pose, [contour], -1, color=(255,0,0), thickness=3)
        img = img_pose

        keypoints_f = front_img_info[front_name]['keypoints']
        landmarks_f = front_img_info[front_name]['landmarks']
        landmarks_s = estimate_side_measures(contour, keypoints_s)
        for name, points in landmarks_s.items():
            cv.line(img, int_tuple(points[0]), int_tuple(points[1]), (0, 255, 255), thickness=LINE_THICKNESS)

        plt.imshow(img[:, :, ::-1])
        plt.savefig(f'{OUT_MEASUREMENT_DIR}{img_path.name}', dpi=1000)
        #plt.show()