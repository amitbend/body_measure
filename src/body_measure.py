import cv2 as cv
import os
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from pathlib import Path
import util as ut
from util import (is_valid_keypoint, is_valid_keypoint_1, pair_dir, int_tuple)
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points
import argparse

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

G_debug_img_s = None
G_debug_img_f = None

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

def radius_search_on_contour(contour, point, radius):
    diffs = contour - point
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    mask = dists <= radius
    return np.where(mask == True)[0], dists[mask]

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

def k_furthest_point_points(point, points, k = 2):
    dsts = linalg.norm(points - point, axis=1)
    sorted_idxs = np.argsort(-dsts)
    return points[sorted_idxs[:k]]

def k_closest_intersections(p, p0, p1, contour_string, k = 1):
    segment = LineString([p0, p1])
    isect_points = segment.intersection(contour_string)
    if isect_points.geom_type == 'Point':
        return np.array([isect_points.x, isect_points.y]).reshape((1,2))
    elif isect_points.geom_type == 'MultiPoint':
        isect_points = np.array([(p.x, p.y) for p in isect_points])
        points = k_closest_point_points(p, isect_points, k = k)
        return np.vstack(points).reshape((k,2))
    else:
        return None

def closet_point_on_contour(contour_str, point):
    min_dst = 9999999999999
    min_idx = -1
    for i, cnt_p in enumerate(contour_str.coords):
        dx = (point[0] - cnt_p[0])
        dy = (point[1] - cnt_p[1])
        dst = dx*dx + dy*dy
        if dst < min_dst:
            min_dst = dst
            min_idx = i
    return min_idx, np.sqrt(min_dst)

def isect_segment_contour(contour, p0, p1):
    a = LineString([p0, p1])
    cnt_points = [contour[i].flatten() for i in range(len(contour))]
    b = LineString(cnt_points)
    ipoints = a.intersection(b)
    return [(p.x, p.y) for p in ipoints]

def extend_segments(p0, p1, percent):
    dir = p0 - p1
    p0_ext = p0 + percent * dir
    p1_ext = p1 - percent * dir
    return p0_ext, p1_ext

def orthor_dir(vec):
    return np.array([vec[1], -vec[0]])

def side_img_is_front_point(contour_str_s, keypoints_s, i):
    nose   = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    midhip = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    p = contour_str_s.coords[i]
    if (p[0] - midhip[0]) * (nose[0] - midhip[0]) > 0 :
        return True
    else:
        return False

def neck_hip_length(keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    hip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    return np.linalg.norm(neck - hip)

def cross_section_points(contour_string, center, line_p0, line_p1, nearest = True):
    segment = LineString([line_p0, line_p1])
    isect_points = segment.intersection(contour_string)
    if isect_points.geom_type == 'MultiPoint':
        isect_points = np.array([(p.x, p.y) for p in isect_points])
        if nearest == True:
            p0, p1 = k_closest_point_points(center, isect_points)
        else:
            p0, p1 = k_furthest_point_points(center, isect_points)
        return True, p0, p1
    else:
        return False, np.zeros(2), np.zeros(2)

def curvature(contour_string, i, half_win_size, point_value_func):
    avg = 0.0
    n_point = len(contour_string.coords)
    for neighbor in (i-half_win_size, i+half_win_size):
        #neighbor = neighbor % n_point
        avg += point_value_func(contour_string.coords[neighbor])
    return 2*point_value_func(contour_string.coords[i]) - avg

def estimate_front_hip(contour_front, keypoints_front):
    lhip_front = keypoints_front[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip_front = keypoints_front[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    lhip_front_ext, rhip_front_ext = extend_segment(lhip_front, rhip_front, 4)
    ipoints = isect_segment_contour(contour_front, lhip_front_ext, rhip_front_ext)
    hip_bdr_front_0 = closest_point_points(lhip_front, ipoints)
    hip_bdr_front_1 = closest_point_points(rhip_front, ipoints)
    return np.vstack([hip_bdr_front_0, hip_bdr_front_1])

def estimate_front_under_midhip_tip_point(contour_string, keypoints):
    lhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    x_range = (min(lhip[0], rhip[0]), max(lhip[0], rhip[0]))
    min_y = 9999
    for p in contour_string.coords:
        if p[0] > x_range[0] and p[0] < x_range[1] and p[1] > lhip[1]:
            if p[1] <  min_y:
                min_y = p[1]
                crotch = p
    return np.array(crotch).flatten(0)

def estimate_front_collar(contour, keypoints):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    half_length = linalg.norm(lshoulder-neck)
    up_dir = normalize(nose - neck)

    lpoint_ext = lshoulder + half_length*up_dir
    rpoint_ext = rshoulder + half_length*up_dir

    contour_string = LineString([contour[i, 0, :] for i in range(contour.shape[0])])
    lcollar =  k_closest_intersections(neck, neck, lpoint_ext, contour_string, k=1)
    rcollar =  k_closest_intersections(neck, neck, rpoint_ext, contour_string, k=1)
    if lcollar.shape[0] == 1 and rcollar.shape[0] == 1:
        collar_0 = lcollar if lcollar[0,1] > rcollar[0,1] else rcollar
        p = nearest_points(Point(collar_0.flatten()), LineString([neck, nose]))[1]
        p = np.array(p.coords[:])
        collar_1 = p + (p - collar_0)
        return np.vstack([collar_0, collar_1])
    else:
        print('estimate_front_collar: two intersection points needed')

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

def estimate_front_crotch(contour_string, keypoints, crotch_ratio_s):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    lhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    lknee = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    lcrotch_pos = lknee + crotch_ratio_s * (lhip - lknee)
    crotch_pos = midhip.copy()
    crotch_pos[1] = lcrotch_pos[1]
    p0 = crotch_pos + (lhip - rhip)
    p1 = crotch_pos - (lhip - rhip)
    # set nearest to false to avoid the two cloest intersection points on the inner profies under crotch
    ret, crotch_0, crotch_1 = cross_section_points(contour_string, crotch_pos, p0, p1, nearest=False)
    if ret == True:
        return True, np.vstack([crotch_0, crotch_1]), crotch_pos
    else:
        return True, np.vstack([np.zeros(2), np.zeros(2)]), crotch_pos

def estimate_front_inside_leg_points(contour, keypoints):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    lankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    rankle = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]
    ankle = 0.5 * (lankle + rankle)
    return np.vstack([midhip, ankle])

def estimate_front_shoulder_points(contour_string, keypoints):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]

    length = linalg.norm(rshoulder- lshoulder)

    up_dir = np.array([0, -1.0])
    p0 = lshoulder
    p1 = p0 + 0.5 * length * (normalize(lshoulder - rshoulder) + up_dir)
    isect_points = LineString([p0, p1]).intersection(contour_string)
    sd_0 = nearest_points(Point(p0), isect_points)[1]
    sd_0 = np.array(sd_0.coords).flatten()

    p0 = rshoulder
    p1 = p0 + 0.5 * length * (normalize(rshoulder - lshoulder) + up_dir)
    isect_points = LineString([p0, p1]).intersection(contour_string)
    sd_1 =  nearest_points(Point(p0), isect_points)[1]
    sd_1 = np.array(sd_1.coords).flatten()

    seg_axis = LineString([midhip, midhip + 2*(neck-midhip)])
    seg_sd = LineString([sd_0, sd_1])
    sd_pos = seg_sd.intersection(seg_axis)
    sd_pos = np.array(sd_pos.coords).flatten()

    min_shoulder_y = min(sd_0[1], sd_1[1])
    sd_0[1] = sd_1[1] = min_shoulder_y

    return np.vstack([sd_0, sd_1]), sd_pos

def estimate_front_armpit_1(contour_str, keypoints):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    lext_shoulder = lshoulder + 0.2*(lshoulder - neck)
    rext_shoulder = rshoulder + 0.2*(rshoulder - neck)

    is_left_neck = lambda p : bool((p[0] - neck[0]) * (lshoulder[0]-neck[0]) > 0)
    xrange_min = min(lext_shoulder[0], rext_shoulder[0])
    xrange_max = max(lext_shoulder[0], rext_shoulder[0])
    lmin_dst = 99999999
    rmin_dst = 99999999
    lclosest = None
    rclosest = None
    for p_cnt in contour_str.coords:
        p_cnt = np.array(p_cnt).flatten()
        #below neck
        if p_cnt[1] > neck[1]:
            if xrange_min < p_cnt[0]  and p_cnt[0] < xrange_max:
                is_left = is_left_neck(p_cnt)
                if is_left:
                    # on left side
                    dst = linalg.norm(p_cnt-neck)
                    if dst < lmin_dst:
                        lmin_dst = dst
                        lclosest = p_cnt
                else:
                    # on right side
                    dst = linalg.norm(p_cnt-neck)
                    if dst < rmin_dst:
                        rmin_dst = dst
                        rclosest = p_cnt

    #cv.drawMarker(G_debug_img_f, (int(lclosest[0]), int(lclosest[1])), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=5, thickness=5)
    #cv.drawMarker(G_debug_img_f, (int(rclosest[0]), int(rclosest[1])), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=5, thickness=5)
    if lclosest is not None and rclosest is not None:
        return np.vstack([lclosest, rclosest])
    else:
        print("estimate_front_armpit: cannot find armpits")
        return np.vstack([lext_shoulder, rext_shoulder])

def estimate_front_armpit(contour, keypoints):
    lelbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['LElbow']][:2]
    relbow = keypoints[POSE_BODY_25_BODY_PART_IDXS['RElbow']][:2]

    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

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

    #take the highest armpit
    larmpit = contour[larmpit_idx, 0, :]
    rarmpit = contour[rarmpit_idx, 0, :]
    #midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    #neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    # if larmpit[1] < rarmpit[1]:
    #     highest_armpit = larmpit
    #     half_length = Point(highest_armpit).distance(LineString([neck, midhip]))
    #     mirror_armpit = highest_armpit + 2 * half_length*normalize(rshoulder - lshoulder)
    # else:
    #     highest_armpit = rarmpit
    #     half_length = Point(highest_armpit).distance(LineString([neck, midhip]))
    #     mirror_armpit = highest_armpit + 2 * half_length*normalize(lshoulder - rshoulder)
    #return np.vstack([highest_armpit, mirror_armpit])

    return np.vstack([larmpit, rarmpit])

def estimate_side_crotch(contour_str, keypoints, hor_dir = np.array([1,0])):
    if is_valid_keypoint_1(keypoints, 'LHip') and is_valid_keypoint_1(keypoints, 'LKnee'):
        hip  = keypoints[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
        knee   = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    elif is_valid_keypoint_1(keypoints, 'RHip') and is_valid_keypoint_1(keypoints, 'RKnee'):
        hip  = keypoints[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
        knee   = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
    else:
        print('estimate_side_crotch: no hip and knee found!')

    lower = hip
    upper = 0.5 * (hip + knee)
    n_point = len(contour_str.coords)
    steepest = 0
    win_size = 5
    for i in range(n_point):
        cnt_p = contour_str.coords[i]
        if lower[1] < cnt_p[1] and cnt_p[1] < upper[1]:
            if not side_img_is_front_point(contour_str, keypoints, i):
                cnt_p_prev = np.array(contour_str.coords[(i-win_size)])
                cnt_p_next = np.array(contour_str.coords[(i+win_size)%n_point])
                dir = normalize(cnt_p_next - cnt_p_prev)
                steep = np.abs(dir.dot(hor_dir))
                if steep > steepest:
                    steepest = steep
                    steepest_cnt_p = cnt_p

    #cv.drawMarker(G_debug_img_s, (int(steepest_cnt_p[0]), int(steepest_cnt_p[1])), (0, 0, 255), markerType=cv.MARKER_SQUARE, markerSize=2,
    #              thickness=1)

    seg = LineString([hip, knee])
    crotch_pos = nearest_points(seg, Point(steepest_cnt_p))[0]
    crotch_pos = np.array([crotch_pos.x, crotch_pos.y])

    hor_seg = 1.5*linalg.norm(hip-knee)*hor_dir
    p0 = crotch_pos + hor_seg
    p1 = crotch_pos - hor_seg

    is_ok, crotch_0, crotch_1 = cross_section_points(contour_str, crotch_pos, p0, p1)
    if is_ok is True:
        ratio = linalg.norm(crotch_pos - knee) / linalg.norm(hip - knee)
        return np.vstack([crotch_0, crotch_1]), crotch_pos, ratio
    else:
        print('estimate_side_crotch: not enough 2 intersection points found')
        return np.vstack([np.zeros(2), np.zeros(2)]), crotch_pos, 0.7
    return np.vstack([lower, upper])

def estimate_side_shoulder(contour_str, keypoints, torso_hor_dir = np.array([1,0])):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    dir = linalg.norm(neck-midhip)*torso_hor_dir
    p0 = neck + dir
    p1 = neck - dir
    ret, sd_0, sd_1 = cross_section_points(contour_str, neck, p0, p1)
    if ret == True:
        return np.vstack([sd_0, sd_1]), neck
    else:
        print('estimate_side_shoulder: cannot find 2 enough intersection points')
        return np.vstack([neck + 0.3*orthor_dir(dir), neck - 0.3*orthor_dir(dir)]), neck

def estimate_side_bust(contour_string, keypoints, torso_hor_dir = np.array([1,0])):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]

    largest_curvature = -1000000000
    upper_bust = midhip + 0.8 * (neck - midhip)
    below_bust = midhip + 0.5 * (neck - midhip)
    half_win_size = 5

    relative_height = lambda point : np.abs(point[0] - neck[0])

    for i in range(len(contour_string.coords)):
        if side_img_is_front_point(contour_string, keypoints, i):
            cnt_p = contour_string.coords[i]
            if upper_bust[1] < cnt_p[1]and cnt_p[1] < below_bust[1]:
                curv = curvature(contour_string, i, half_win_size, relative_height)
                if curv > largest_curvature:
                    furthest_slice_pos = cnt_p
                    largest_curvature = curv

    seg = LineString([neck, midhip])
    bust_pos = nearest_points(seg, Point(furthest_slice_pos))[0]
    bust_pos = np.array([bust_pos.x, bust_pos.y])

    hor_seg = linalg.norm(neck-midhip) * torso_hor_dir
    p0 = bust_pos + hor_seg
    p1 = bust_pos - hor_seg

    is_ok, under_bust_0, under_bust_1 = cross_section_points(contour_string, bust_pos, p0, p1)
    if is_ok is True:
        return np.vstack([under_bust_0, under_bust_1]), bust_pos
    else:
        print('estimate_side_under_bust: not enough 2 intersection points found')
        return np.vstack([np.zeros(2), np.zeros(2)]), bust_pos

def estimate_side_armscye(contour_str, bust, shoulder, torso_hor_dir = np.array([1,0])):
    armscye_pos = bust + 1.0/3.0 * (shoulder - bust)
    hor_ext_seg = linalg.norm(shoulder-bust)*5*torso_hor_dir
    p0 = armscye_pos + hor_ext_seg
    p1 = armscye_pos - hor_ext_seg
    ret, sd_0, sd_1 = cross_section_points(contour_str, armscye_pos, p0, p1)
    if ret == True:
        return np.vstack([sd_0, sd_1]), armscye_pos
    else:
        print('estimate_side_armscye: cannot find 2 enough intersection points')
        len = 0.5*linalg.norm(shoulder-bust)
        return np.vstack([armscye_pos + len*torso_hor_dir, armscye_pos - len*torso_hor_dir]), armscye_pos

def estimate_side_under_bust(contour_string, keypoints, bust_pos, torso_hor_dir = np.array([1,0])):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    hor_seg = linalg.norm(neck-midhip)*torso_hor_dir

    below_under_bust = midhip + 0.333 * (neck - midhip)

    half_win_size = 6
    relative_height = lambda point : np.abs(point[0] - neck[0])

    smallest_curv = 999999999
    for i in range(len(contour_string.coords)):
        if side_img_is_front_point(contour_string, keypoints, i):
            cnt_p = contour_string.coords[i]
            if bust_pos[1] < cnt_p[1]and cnt_p[1] < below_under_bust[1]:
                curv = curvature(contour_string, i, half_win_size, relative_height)
                if curv < smallest_curv:
                    furthest_slice_pos = cnt_p
                    smallest_curv = curv

    under_bust_p = nearest_points(LineString([midhip, neck]), Point(furthest_slice_pos))[0]
    under_bust_p = np.array([under_bust_p.x, under_bust_p.y])
    p0 = under_bust_p + hor_seg
    p1 = under_bust_p - hor_seg
    is_ok, under_bust_0, under_bust_1 = cross_section_points(contour_string, under_bust_p, p0, p1)
    if is_ok is True:
        return np.vstack([under_bust_0, under_bust_1]), under_bust_p
    else:
        print('estimate_side_under_bust: not enough 2 intersection points found')
        return np.vstack([np.zeros(2), np.zeros(2)]), under_bust_p

def estimate_front_height(contour_str, keypoints_f):
    lowest = 99999
    for p in contour_str.coords:
        if p[1] < lowest:
            lowest = p[1]
            head_tip = p

    lhip = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    lknee  = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    lankle = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
    lheel = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LHeel']][:2]

    on_hip_level = nearest_points(Point(head_tip), LineString([lhip, rhip]))[1]
    on_hip_level = np.array(on_hip_level ).flatten()

    height = linalg.norm(on_hip_level - head_tip)
    height += linalg.norm(lknee - lhip)
    height += linalg.norm(lankle - lknee)
    height += linalg.norm(lheel - lankle)

    return np.vstack([head_tip, head_tip+height*np.array([0,1])])
    #return np.vstack([head_tip, on_hip_level])

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
        #return np.vstack([head_tip, bottom])
        height = linalg.norm(bottom - head_tip)
        return np.vstack([head_tip, head_tip + height * np.array([0, 1])])
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

        height = linalg.norm(bottom - head_tip)
        return np.vstack([head_tip, head_tip + height * np.array([0, 1])])
        #return np.vstack([head_tip, bottom])

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

def estimate_front_collar_bust(collar, bust):
    collar_0 = collar[0,:]
    proj_on_bust = nearest_points(Point(collar_0), LineString([bust[0,:], bust[1,:]]))[1]
    proj_on_bust = np.array(proj_on_bust.coords[:])
    return np.vstack([collar_0, proj_on_bust])

def estimate_front_collar_waist(collar, waist):
    collar_0 = collar[1,:]
    proj_on_waist = nearest_points(Point(collar_0), LineString([waist[0,:], waist[1,:]]))[1]
    proj_on_waist = np.array(proj_on_waist.coords[:])
    return np.vstack([collar_0, proj_on_waist])

def extract_chain(contour_str, p0, p1):
    idx_0, _  = closet_point_on_contour(contour_str, p0)
    idx_1, _  = closet_point_on_contour(contour_str, p1)
    if idx_0 < idx_1:
        return contour_str.coords[idx_0:idx_1] + [contour_str.coords[idx_1]]
    else:
        return list(reversed(contour_str.coords[idx_1:idx_0] + [contour_str.coords[idx_0]]))

def extract_torso_contour(contour_str_f, keypoints_f, armpit_l, armpit_r):
    lhip =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    rhip =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
    lshouder =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshouder=  keypoints_f[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]

    lchain = extract_chain(contour_str_f, lhip, armpit_l)
    lchain.append(lshouder)

    rchain = extract_chain(contour_str_f, rhip, armpit_r)
    rchain.append(rshouder)
    torso = lchain + list(reversed(rchain))

    #for point in rchain:
    #    cv.drawMarker(G_debug_img_f, (int(point[0]), int(point[1])), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=3)

    #cv.drawMarker(G_debug_img_f, (int(lhip[0]), int(lhip[1])), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=3)
    #cv.drawMarker(G_debug_img_f, (int(armpit_l[0]), int(armpit_l[1])), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=3)

    return LineString(torso)

def extract_leg_contour(contour_str_f, keypoints_f, crotch_pos, left = True):
    if left:
        hip =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
    else:
        hip =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]

    crotch_tip = estimate_front_under_midhip_tip_point(contour_str_f, keypoints_f)
    leg_points = extract_chain(contour_str_f, hip, crotch_tip)
    leg_points.append(crotch_pos)

    #for point in chain:
    #    cv.drawMarker(G_debug_img_f, (int(point[0]), int(point[1])), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=3)

    return LineString(leg_points)

# note: axis should be long enough to cover the largest hor slice of torso
def calc_contour_slice(contour_str, pos, hor_seg):
    p0 = pos + hor_seg
    p1 = pos - hor_seg
    ret, slc_p0, slc_p1 = cross_section_points(contour_str, pos, p0, p1, nearest=True)
    if ret == True:
        return True, np.vstack([slc_p0, slc_p1])
    else:
        return False, np.vstack([p0, p1])

def estimate_front_leg_landmark_slices(contour_str_f, keypoints_f, crotch_pos_f, left_side, landmarks_f):
    contour_leg_str_f =  extract_leg_contour(contour_str_f, keypoints_f, crotch_pos_f, left=left_side)

    if left_side: prefix = 'L'
    else: prefix = 'R'

    hip =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS[prefix  +'Hip']][:2]
    knee =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS[prefix +'Knee']][:2]
    ankle =  keypoints_f[POSE_BODY_25_BODY_PART_IDXS[prefix+'Ankle']][:2]
    #dir_hip_knee = normalize(hip - knee)
    #dir_knee_ankle = normalize(knee - ankle)
    leg_hor_dir = np.array([1,0])
    ext_len = linalg.norm(hip-knee)

    crotch_pos_on_leg_f = nearest_points(Point(crotch_pos_f), LineString([hip, knee]))[1]
    crotch_pos_on_leg_f = np.array(crotch_pos_on_leg_f).flatten()

    ret, points = calc_contour_slice(contour_leg_str_f, knee, ext_len*leg_hor_dir)
    if ret: landmarks_f[prefix + 'Knee'] = points

    pos = 0.5*(crotch_pos_on_leg_f+knee)
    ret, points = calc_contour_slice(contour_leg_str_f, pos, ext_len*leg_hor_dir)
    if ret: landmarks_f['Aux_' + prefix + 'Thigh'] = points

    pos = ankle + 2.0/3.0 *(knee - ankle)
    ret, points = calc_contour_slice(contour_leg_str_f, pos, ext_len*leg_hor_dir)
    if ret: landmarks_f[prefix + 'BigUnderLeg'] = points

    #move anke a bit higher to avoid perspective distortion effect; otherwise, we will measure cross section of the sandle
    pos = ankle + 0.1 * (knee - ankle)
    ret, points = calc_contour_slice(contour_leg_str_f, pos, ext_len*leg_hor_dir)
    if ret: landmarks_f[prefix + 'Ankle'] = points

def estimate_landmark_slices(contour_f, keypoints_f, contour_s, keypoints_s):
    landmarks_f = {}
    landmarks_s = {}

    contour_str_s = LineString([contour_s[i,0,:] for i in range(contour_s.shape[0])])
    contour_str_f = LineString([contour_f[i,0,:] for i in range(contour_f.shape[0])])

    ###############################################################################################
    #side image
    neck_s = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip_s = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]

    if is_valid_keypoint_1(keypoints_s, 'LHip') and is_valid_keypoint_1(keypoints_s, 'LKnee'):
        hip = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['LHip']][:2]
        knee = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    else:
        hip = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['RHip']][:2]
        knee = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]

    torso_hor_seg_s = linalg.norm(neck_s - midhip_s)*np.array([1,0])
    thigh_hor_seg_s = linalg.norm(knee - hip)*np.array([1,0])

    points = estimate_side_height(contour_s, keypoints_s)
    landmarks_s['Height'] = points

    points, shoulder_pos_s = estimate_side_shoulder(contour_str_s, keypoints_s)
    landmarks_s['Shoulder'] = points

    points, bust_pos_s = estimate_side_bust(contour_str_s, keypoints_s)
    landmarks_s['Bust'] = points

    points, armscye_pos_s = estimate_side_armscye(contour_str_s, bust_pos_s, shoulder_pos_s)
    landmarks_s['Armscye'] = points

    points, under_bust_pos_s= estimate_side_under_bust(contour_str_s, keypoints_s, bust_pos_s)
    landmarks_s['UnderBust'] = points

    is_ok, points =  calc_contour_slice(contour_str_s, hip, 2*torso_hor_seg_s)
    hip_pos_s = hip.copy()
    if is_ok: landmarks_s['Hip'] = points

    points, crotch_pos_s, crotch_ratio_s = estimate_side_crotch(contour_str_s, keypoints_s)
    landmarks_s['Crotch'] = points

    waist_pos_s =  0.5*(crotch_pos_s + bust_pos_s)
    #points, waist_pos_s = estimate_side_waist(contour_str_s, bust_pos_s, crotch_pos_s)
    is_ok, points = calc_contour_slice(contour_str_s, waist_pos_s, torso_hor_seg_s)
    if is_ok: landmarks_s['Waist'] = points

    hip_waist_pos_0_s = 0.5*(hip_pos_s + waist_pos_s)
    is_ok, points = calc_contour_slice(contour_str_s, hip_waist_pos_0_s, torso_hor_seg_s)
    if is_ok: landmarks_s['Aux_Hip_Waist_0'] = points

    waist_under_bust_pos_0_s = 0.5*(under_bust_pos_s + waist_pos_s)
    is_ok, points =  calc_contour_slice(contour_str_s, waist_under_bust_pos_0_s, torso_hor_seg_s)
    if is_ok: landmarks_s['Aux_Waist_UnderBust_0'] = points

    armcye_shoulder_0_s = 0.5*(armscye_pos_s +  shoulder_pos_s)
    is_ok, points =  calc_contour_slice(contour_str_s, armcye_shoulder_0_s, torso_hor_seg_s)
    if is_ok: landmarks_s['Aux_Armscye_Shoulder_0'] = points

    aux_crotch_hip_0_s = 0.5*(crotch_pos_s+  hip_pos_s)
    is_ok, points =  calc_contour_slice(contour_str_s, aux_crotch_hip_0_s, thigh_hor_seg_s)
    if is_ok: landmarks_s['Aux_Crotch_Hip_0'] = points

    pos = 0.5*(knee+crotch_pos_s)
    ret, points = calc_contour_slice(contour_str_s, pos, thigh_hor_seg_s)
    if ret: landmarks_s['Aux_Thigh_0'] = points

    is_ok, points =  calc_contour_slice(contour_str_s, knee, thigh_hor_seg_s)
    if is_ok: landmarks_s['Knee'] = points

    ###############################################################################################
    #front image
    neck_f = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip_f = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    lshouder_f = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshouder_f = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    torso_hor_seg_f = 2*linalg.norm(lshouder_f - rshouder_f)*np.array([1,0])

    # height
    points = estimate_front_height(contour_str_f, keypoints_f)
    landmarks_f['Height'] = points

    # neck
    points = estimate_front_neck_points(contour_f, keypoints_f)
    landmarks_f['Neck'] = points

    # shoulder
    points, shoulder_pos_f = estimate_front_shoulder_points(contour_str_f, keypoints_f)
    landmarks_f['Shoulder'] = points

    # armpit
    armpits_f = estimate_front_armpit_1(contour_str_f, keypoints_f)
    armpit_l_f = armpits_f[0]
    armpit_r_f = armpits_f[1]

    # crotch
    ret, points, crotch_pos_f = estimate_front_crotch(contour_str_f, keypoints_f, crotch_ratio_s)
    if ret: landmarks_f['Crotch'] = points

    ###################################################################
    # torso extraction
    torso_str_f = extract_torso_contour(contour_str_f, keypoints_f, armpit_l_f, armpit_r_f)

    # under bust
    # regress under bust position from the under bust postioin ration in the side image
    under_bust_ratio_s = linalg.norm((under_bust_pos_s - midhip_s)) / linalg.norm(neck_s - midhip_s)
    # it seems that neck point in side image is equavalent to shoulder point in front image, not the neck point in the front image
    under_bust_pos_f = midhip_f + under_bust_ratio_s * (shoulder_pos_f - midhip_f)
    is_ok, points =  calc_contour_slice(torso_str_f, under_bust_pos_f, torso_hor_seg_f)
    if is_ok: landmarks_f['UnderBust'] = points

    # bust
    bust_ratio_s = linalg.norm((bust_pos_s - midhip_s)) / linalg.norm(neck_s - midhip_s)
    bust_pos_f = midhip_f + bust_ratio_s * (shoulder_pos_f - midhip_f)
    is_ok, points =  calc_contour_slice(torso_str_f, bust_pos_f, torso_hor_seg_f)
    if is_ok: landmarks_f['Bust'] = points

    armscye_ratio_s = linalg.norm((armscye_pos_s - midhip_s)) / linalg.norm(neck_s - midhip_s)
    armcye_pos_f = midhip_f + armscye_ratio_s * (shoulder_pos_f - midhip_f)
    is_ok, points =  calc_contour_slice(torso_str_f, armcye_pos_f, torso_hor_seg_f)
    landmarks_f['Armscye'] = points

    #waist
    waist_pos_f = 0.5*(crotch_pos_f + bust_pos_f)
    is_ok, points =  calc_contour_slice(torso_str_f, waist_pos_f, torso_hor_seg_f)
    if is_ok: landmarks_f['Waist'] = points

    # auxilary slice on torso
    hip_waist_pos_0 = 0.5*(midhip_f+ waist_pos_f)
    is_ok, points =  calc_contour_slice(torso_str_f, hip_waist_pos_0, torso_hor_seg_f)
    if is_ok: landmarks_f['Aux_Hip_Waist_0'] = points

    waist_under_bust_pos_0 = 0.5*(under_bust_pos_f + waist_pos_f)
    is_ok, points =  calc_contour_slice(torso_str_f, waist_under_bust_pos_0, torso_hor_seg_f)
    if is_ok: landmarks_f['Aux_Waist_UnderBust_0'] = points

    armscye_shoulder_0 = 0.5*(armcye_pos_f +  shoulder_pos_f)
    is_ok, points =  calc_contour_slice(contour_str_f, armscye_shoulder_0, torso_hor_seg_f)
    if is_ok: landmarks_f['Aux_Armscye_Shoulder_0'] = points

    aux_crotch_hip_0_f = 0.5*(crotch_pos_f+  midhip_f)
    is_ok, points =  calc_contour_slice(contour_str_f, aux_crotch_hip_0_f, torso_hor_seg_f)
    if is_ok: landmarks_f['Aux_Crotch_Hip_0'] = points

    # collar
    points = estimate_front_collar(contour_f, keypoints_f)
    landmarks_f['Collar'] = points

    # collar - bust
    points = estimate_front_collar_bust(landmarks_f['Collar'], landmarks_f['Bust'] )
    landmarks_f['CollarBust'] = points

    # hip
    points = estimate_front_hip(contour_f, keypoints_f)
    landmarks_f['Hip'] = points

    # collar - waist
    points = estimate_front_collar_waist(landmarks_f['Collar'], landmarks_f['Waist'])
    landmarks_f['CollarWaist'] = points

    ##############################################
    # arm
    # left elbow
    points = estimate_front_elbow(contour_f, keypoints_f, left=False)
    landmarks_f['LElbow'] = points

    #wrist
    points = estimate_front_wrist(contour_f, keypoints_f, left=False)
    landmarks_f['LWrist'] = points

    ##########################################################
    # inside leg
    points = estimate_front_inside_leg_points(contour_f, keypoints_f)
    landmarks_f['InsideLeg'] = points

    ##########################################################
    estimate_front_leg_landmark_slices(contour_str_f, keypoints_f, crotch_pos_f, left_side=True, landmarks_f=landmarks_f)
    #estimate_leg_landmark_slices(contour_str_f, keypoints_f, crotch_pos_f, left_side=False, landmarks_f=landmarks_f)

    return landmarks_f, landmarks_s

def extend_rect(rect, percent, img_shape):
    top_left_x = int(max(rect[0] - 0.5*percent[0]*rect[2], 0))
    top_left_y = int(max(rect[1] - 0.5*percent[1]*rect[3], 0))
    bot_right_x = int(min(rect[0] + rect[2] + 0.5*percent[0]*rect[2], img_shape[1]))
    bot_right_y = int(min(rect[1] + rect[3] + 0.5*percent[1]*rect[3], img_shape[0]))
    return (top_left_x, top_left_y, bot_right_x-top_left_x, bot_right_y-top_left_y)

#pos_t: target position in image
#pos_o: origin position in sub_image
def embed_img_img(img, pos_t, sub_img, pos_o):
    x = pos_t[0] - pos_o[0]
    y = pos_t[1] - pos_o[1]
    assert x >= 0 and y >= 0
    w = sub_img.shape[1]
    h = sub_img.shape[0]
    img[y:y+h, x:x+w, :] = sub_img
    return img

def align_front_side_img(img_f, landmarks_f, keypoints_f, img_s, landmarks_s, keypoints_s):
    #hack y_offset. need to large enough to contain two images
    y_offset = int(0.1 * img_f.shape[0])

    width_ext  = img_f.shape[1] + img_s.shape[1]
    height_ext = img_f.shape[0] + 2*y_offset

    img_f_s = np.zeros((height_ext, width_ext, 3), dtype=np.uint8)

    neck_f = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2].astype(np.int32)
    neck_s = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2].astype(np.int32)
    midhip_f = keypoints_f[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2].astype(np.int32)
    midhip_s = keypoints_s[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2].astype(np.int32)

    #anchor_0_f = landmarks_f['Height'][0].astype(np.int32)
    anchor_0_f = neck_f
    anchor_0_embed_f = anchor_0_f.copy()
    anchor_0_embed_f[1] += y_offset

    img_f_s = embed_img_img(img_f_s, anchor_0_embed_f, img_f, anchor_0_f)

    #anchor_0_s = landmarks_s['Height'][0].astype(np.int32)
    anchor_0_s = neck_s
    anchor_0_embed_s = anchor_0_s.copy()
    anchor_0_embed_s[0] = anchor_0_s[0] + img_f.shape[1]
    anchor_0_embed_s[1] = anchor_0_embed_f[1]
    img_f_s = embed_img_img(img_f_s, anchor_0_embed_s, img_s, anchor_0_s)

    cv.line(img_f_s, int_tuple(anchor_0_embed_f), int_tuple(anchor_0_embed_s), color=(0, 0, 255), thickness=5)

    #anchor_1_f = landmarks_f['Height'][1].astype(np.int32)
    anchor_1_f = midhip_f
    anchor_1_embed_f = anchor_0_embed_f + (anchor_1_f - anchor_0_f)

    #anchor_1_s = landmarks_s['Height'][1].astype(np.int32)
    anchor_1_s = midhip_s
    anchor_1_embed_s = anchor_0_embed_s + (anchor_1_s - anchor_0_s)
    cv.line(img_f_s, int_tuple(anchor_1_embed_f), int_tuple(anchor_1_embed_s), color=(0, 0, 255), thickness=5)

    return img_f_s

def resize_img_to_fit_silhouette(img, sil):
    contour = ut.find_largest_contour(sil)

    bb_sil = cv.boundingRect(contour)
    bb_sil = extend_rect(bb_sil, (0.2, 0.05), img.shape)

    img = img[bb_sil[1]:bb_sil[1] + bb_sil[3], bb_sil[0]:bb_sil[0] + bb_sil[2], :].copy()
    sil = sil[bb_sil[1]:bb_sil[1] + bb_sil[3], bb_sil[0]:bb_sil[0] + bb_sil[2]]

    return img, sil

import sys
def load_silhouette(path, img):
    sil = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if sil is None:
        print(f'silhouette {path} is not found', file=sys.stderr)
    sil = cv.resize(sil, (img.shape[1], img.shape[0]), cv.INTER_NEAREST)
    ret, sil = cv.threshold(sil, 200, maxval=255, type=cv.THRESH_BINARY)
    return sil

def fix_silhouette(sil):
    sil = cv.morphologyEx(sil, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, ksize=(3,3)))
    return sil

def normalize_measurement_based_on_height(height, landmarks_f, landmarks_s):
    height_points_f = landmarks_f['Height']
    height_points_s = landmarks_s['Height']
    height_px_f = linalg.norm(height_points_f[0] - height_points_f[1])
    height_px_s = linalg.norm(height_points_s[0] - height_points_s[1])
    ratio_f = height/height_px_f
    ratio_s = height/height_px_s

    measure_f = {}
    for id, points in landmarks_f.items():
        if id is not 'Height':
            measure_f[id] = ratio_f * linalg.norm(points[0]-points[1])

    measure_s = {}
    for id, points in landmarks_s.items():
        if id is not 'Height':
            measure_s[id] = ratio_s * linalg.norm(points[0]-points[1])

    return measure_f, measure_s

def calc_body_slices(sil_f, sil_s,  keypoints_f, keypoints_s):
    contour_f = ut.find_largest_contour(sil_f, app_type=cv.CHAIN_APPROX_NONE)
    contour_f = ut.smooth_contour(contour_f, 10)
    contour_f = ut.resample_contour(contour_f, 720)

    # load and process side images
    #img_s, sil_s = resize_img_to_fit_silhouette(img_s, sil_s)
    contour_s = ut.find_largest_contour(sil_s, app_type=cv.CHAIN_APPROX_NONE)
    contour_s = ut.smooth_contour(contour_s, 10)
    contour_s = ut.resample_contour(contour_s, 720)

    slices_f, slices_s = estimate_landmark_slices(contour_f, keypoints_f[0, :, :], contour_s, keypoints_s[0, :, :])

    return contour_f, contour_s, slices_f, slices_s

def draw_slice_data(img, contour, slices, LINE_THICKNESS = 2):
    cv.drawContours(img, [contour], -1, color=(255, 0, 0), thickness=1)
    for i in range(contour.shape[0]):
        cv.drawMarker(img, int_tuple(contour[i, 0, :]), color=(0, 0, 255), markerType=cv.MARKER_SQUARE, markerSize=1, thickness=1)

    for name, points in slices.items():
        if 'Aux_' not in name:
            cv.line(img, int_tuple(points[0]), int_tuple(points[1]), (0, 0, 255), thickness=LINE_THICKNESS)
    #
    for name, points in slices.items():
        if 'Aux_' in name:
            cv.line(img, int_tuple(points[0]), int_tuple(points[1]), (255, 0, 0), thickness=LINE_THICKNESS)

    cv.line(img, int_tuple(slices['Height'][0]), int_tuple(slices['Height'][1]), (0, 255, 255), thickness=LINE_THICKNESS + 2)

def calc_body_slices_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height):
    #MARKER_SIZE = 5
    #MARKER_THICKNESS = 5
    LINE_THICKNESS = 2

    contour_f, contour_s, slices_f, slices_s = calc_body_slices(sil_f, sil_s, keypoints_f, keypoints_s)

    #data = {'contour_f': contour_f, 'contour_s': contour_s}
    #np.save(f'{OUT_DIR}/{path_f.stem}_contour.npy', data)

    #data = {'slices_f': slices_f, 'slices_s': slices_s}
    #np.save(f'{OUT_DIR}/{path_f.stem}_slice.npy', data)

    measure_f, measure_s = normalize_measurement_based_on_height(height, slices_f, slices_s)
    #data = {'measure_f': measure_f, 'measure_s': measure_s}
    #np.save(f'{OUT_DIR}/{path_f.stem}_measure.npy', data)

    data = {'contour_f': contour_f, 'contour_s': contour_s,
            'slices_f' : slices_f, 'slices_s': slices_s,
            'measure_f': measure_f, 'measure_s': measure_s}

    #img_pose_f = cv.imread(f'{POSE_DIR}/{path_f.stem}.png')
    #img_pose_s = cv.imread(f'{POSE_DIR}/{path_s.stem}.png')
    img_pose_f = img_f.copy()
    img_pose_s = img_s.copy()

    draw_slice_data(img_pose_f, contour_f, slices_f)
    draw_slice_data(img_pose_s, contour_s, slices_s)

    # plt.clf()
    # plt.subplot(121), plt.imshow(img_pose_f[:, :, ::-1])
    # plt.subplot(122), plt.imshow(img_pose_s[:, :, ::-1])
    # plt.savefig(f'{OUT_DIR}{path_f.stem}_viz.png', dpi=1000)
    # plt.show()

    final_vis = np.concatenate((img_pose_f, img_pose_s), axis=1)

    return data, final_vis


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="image folder")
    ap.add_argument("-s", "--sil_dir", required=True,   help="silhouette folder")
    ap.add_argument("-p", "--pose_dir", required=True,  help="pose folder")
    ap.add_argument("-o", "--output_dir", required=True, help='output silhouette dir')
    args = vars(ap.parse_args())

    IMG_DIR = args['image_dir'] + '/'
    SIL_DIR = args['sil_dir'] + '/'
    POSE_DIR = args['pose_dir'] + '/'
    OUT_DIR = args['output_dir'] + '/'

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


    for f in Path(OUT_DIR).glob('*.*'):
        os.remove(f)

    mapping = {}
    mapping['side_IMG_1935.JPG'] = 'front_IMG_1928.JPG'
    mapping['side_IMG_1941.JPG'] = 'front_IMG_1939.JPG'
    mapping['side_8E2593C4.jpg'] = 'front_9EF020C7.jpg'

    mapping_inv = {value:key for key, value in mapping.items()}

    heights = {}
    heights['IMG_1928__.JPG'] =  150
    heights['IMG_1939__.JPG'] = 160
    heights['8E2593C4__.jpg'] = 158

    #collect front and side image pairs
    all_img_paths = [path for path in Path(IMG_DIR).glob('*.*')]
    img_f_paths = {}
    img_s_paths = {}
    for path in Path(IMG_DIR).glob('*.*'):
        if 'front_' in str(path):
            id = str(path).replace('front_','')
            img_f_paths[id] = path
        elif 'side_' in str(path):
            id = str(path).replace('side_','')
            img_s_paths[id] = path
        else:
            print(f'{path} is not recognized as a front or side image')

    path_pairs = []
    for id, path_f in img_f_paths.items():
        if id in img_s_paths:
            path_pairs.append((path_f, img_s_paths[id]))

    if len(path_pairs) == 0:
        print('not front and side image pair is found. please change their names appropriately: expected format some_id_front_.jpg, some_id_side_.jpg')

    for path_f, path_s in path_pairs:
        if path_f.name in heights:
            height = heights[path_f.name]
        else:
            height = 170

        img_f = cv.imread(str(path_f))
        keypoints_f = np.load(f'{POSE_DIR}/{path_f.stem}.npy')
        sil_f = load_silhouette(f'{SIL_DIR}{path_f.name}', img_f)

        img_s = cv.imread(str(path_s))
        sil_s = load_silhouette(f'{SIL_DIR}{path_s.name}', img_s)
        keypoints_s = np.load(f'{POSE_DIR}/{path_s.stem}.npy')

        # plt.subplot(121)
        # plt.imshow(img_f)
        # plt.imshow(sil_f, alpha=0.4)
        # plt.subplot(122)
        # plt.imshow(img_s)
        # plt.imshow(sil_s, alpha=0.4)
        # plt.show()

        data, final_viz = calc_body_slices_util(img_f, img_s, sil_f, sil_s, keypoints_f, keypoints_s, height = height)
        cv.imwrite(f'{OUT_DIR}/{path_f.name}', final_viz)

        np.save(f'{OUT_DIR}/{path_f.stem}.npy', data)

