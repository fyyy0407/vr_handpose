"""
Utilities.
"""

import os
import cv2
import logging
import numpy as np

from cv2 import aruco
from transforms3d.axangles import axangle2mat
from easyrobot.utils.logger import ColoredLogger
from transforms3d.quaternions import mat2quat, quat2mat

from teledata.constants import INHAND_TCP_TRANSFORMATION, INHAND_SERIAL


def resolve_logger(name):
    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def pose_mat2quat(pose):
    """
    Transform 4x4 pose to 7-dim xyz + quaternion pose.
    """
    return np.concatenate([pose[:3, 3], mat2quat(pose[:3, :3])], axis = 0)

def pose_quat2mat(pose):
    """
    Transform 7-dim xyz + quaternion pose to 4x4 pose.
    """
    res = np.zeros((4, 4), dtype = np.float32)
    res[:3, :3] = quat2mat(pose[3:])
    res[:3, 3] = pose[:3]
    res[3, 3] = 1
    return res

def retrieve_calib(calib_dir):
    """
    Retrive the most recent calibration result from the calibration folder.
    """
    # Find the most recent calibration record.
    calib_target = -np.inf
    for calib_record in os.listdir(calib_dir):
        try:
            calib_timestamp = int(calib_record)
        except Exception:
            continue
        
        if calib_timestamp > calib_target:
            calib_target = calib_timestamp
    
    assert calib_target > 0, "Cannot find valid calibration records in {}.".format(calib_dir)

    # Fetch calibration informations.
    calib_folder = os.path.join(calib_dir, str(calib_target))

    intrinsics = np.load(os.path.join(calib_folder, "intrinsics.npz"))
    extrinsics = np.load(os.path.join(calib_folder, "extrinsics.npz"))
    tcp = pose_quat2mat(np.load(os.path.join(calib_folder, "tcp.npy")))

    return intrinsics, extrinsics, tcp


def calc_calib(calib_dir, cfg_id, serial, is_global):
    """
    Calculate the calibration results.
    """
    extrinsic = {"is_global": is_global}

    # Retrieve calibration data
    intrinsics, extrinsics, tcp = retrieve_calib(calib_dir)
    assert serial in intrinsics.keys(), "Cannot find camera {} in calibration folder. Please re-calibrate the cameras.".format(serial)

    if serial == INHAND_SERIAL[cfg_id]:
        extrinsic["pose_in_link"] = pose_mat2quat(np.linalg.inv(INHAND_TCP_TRANSFORMATION[cfg_id])).tolist()
        extrinsic["parent_link_name"] = "tcp"
    else:
        if is_global:
            extrinsic["pose_in_link"] = pose_mat2quat(
                tcp @ np.linalg.inv(INHAND_TCP_TRANSFORMATION[cfg_id]) @ extrinsics[INHAND_SERIAL[cfg_id]] @ np.linalg.inv(extrinsics[serial])
            ).tolist()
            extrinsic["parent_link_name"] = "base"
        else:
            extrinsic["pose_in_link"] = pose_mat2quat(
                np.linalg.inv(INHAND_TCP_TRANSFORMATION[cfg_id]) @ extrinsics[INHAND_SERIAL[cfg_id]] @ np.linalg.inv(extrinsics[serial])
            )
            extrinsic["parent_link_name"] = "tcp"
    
    return intrinsics[serial], extrinsic


def aruco_detector(
    img,
    aruco_dict,
    marker_length,
    camera_intrinsic,
    vis = True
):
    """
    Args:
    - img: image in BGR format;
    - aruco_dict: aruco dict config;
    - camera_intrinsic: camera intrinsics;
    - vis: whether to enable detection visualization.

    Returns: a dict includes all detected aruco marker pose.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict)
    aruco_params = aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    aruco_params.cornerRefinementWinSize = 5
    dist_coeffs = np.array([[0., 0., 0., 0.]])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(
        img_gray,
        aruco_dict,
        parameters = aruco_params
    )

    if ids is None:
        if vis:
            cv2.imshow("Detected markers", img)
            cv2.waitKey(0)
        return {}
    
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_intrinsic,
        dist_coeffs
    )

    matrices = {}
    for idx, (t, r) in enumerate(zip(tvec, rvec)):
        mat = np.zeros((4, 4), dtype = np.float32)
        mat[:3, 3] = np.array(t).reshape(3) / 1000
        rotvec = np.array(r).reshape(3)
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle != 0 else np.array([1, 0, 0])
        mat[:3, :3] = axangle2mat(axis, angle)
        mat[3, 3] = 1
        aruco_idx = ids[idx][0]
        matrices[aruco_idx] = mat
    
    if vis:
        draw_img = aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
        for (t, r) in zip(tvec, rvec):
            draw_img = aruco.drawAxis(draw_img, camera_intrinsic, dist_coeffs, r, t, 100)
        cv2.imshow("Detected markers", draw_img)
        cv2.waitKey(0)

    return matrices
