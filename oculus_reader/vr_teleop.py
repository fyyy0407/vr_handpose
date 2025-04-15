"""
Teleoperation Script.
"""

import os
import json
import time
import hydra
import logging
import tempfile
import subprocess
import numpy as np
import multiprocessing

from pynput import keyboard
from omegaconf import OmegaConf
from easyrobot.utils.logger import ColoredLogger

from .teledata.constants import *
from .teledata.utils import resolve_logger
from .teledata.converter import PoseConverter


loop = True
state = 0
finish_time = 0
finish_rating = 0

def _on_press(key):
    global loop, state, finish_time, finish_rating
    try:
        if key.char == 'q':
            loop = False
            return False
        if key.char == 's':
            if state == 0:
                state = 1
                finish_time = int(time.time() * 1000)
        if str.isdigit(key.char):
            if state == 1:
                state = 2
                finish_rating = int(key.char)
    except AttributeError:
        pass
    
def _on_release(key):
    pass


@hydra.main(
    version_base = None,
    config_path = os.path.join("configs")
)
def main(cfg):
    OmegaConf.resolve(cfg)

    # Setup logger
    logger = resolve_logger(cfg.logger.teleop)

    # Initialize robot with gripper
    logger.info("Initialize robot and gripper ...")
    robot = hydra.utils.instantiate(
        cfg.hardware.robot, 
        shm_name = cfg.shm_name.robot
    )
    gripper = hydra.utils.instantiate(
        cfg.hardware.gripper, 
        shm_name = cfg.shm_name.gripper
    )
    gripper_quantization_size = gripper.max_width / cfg.teleop.gripper_quantization_steps

    # Generate robot init pose (also considering random initialization)
    robot_init_pose = np.array(cfg.pose.init.robot)
    if cfg.pose.init.random_xyz:
        for axis in range(3):
            robot_init_pose[axis] += np.random.uniform(*cfg.pose.init.random_xyz_range[axis])

    # Go to the initial pose
    robot.send_tcp_pose(robot_init_pose)
    assert cfg.pose.init.gripper in ["open", "close"]
    if cfg.pose.init.gripper == "open":
        gripper.open_gripper()
    else:
        gripper.close_gripper()
    robot.calib_sensor()
    
    # Initialize oculus_reader
    logger.info("Initialize oculus_reader ...")
    oculus_reader = hydra.utils.instantiate(cfg.hardware.oculus_reader)

    # Initialize camera, assuming that the first global camera is the main camera.
    main_serial = cfg.hardware.camera.main
    logger.info("Initialize main camera {} ...".format(main_serial))
    assert main_serial in (list(cfg.hardware.camera["global"].keys()) + list(cfg.hardware.camera["inhand"].keys())), \
        "Cannot find main camera {} in camera list.".format(main_serial)
    main_camera = hydra.utils.instantiate(
        cfg.hardware.camera["global" if main_serial in cfg.hardware.camera["global"].keys() else "inhand"][main_serial], 
        serial = main_serial,
        shm_name_rgb = cfg.shm_name.main_camera_color,
        shm_name_depth = cfg.shm_name.main_camera_depth
    )
    
    # Initialize pose converter
    converter = PoseConverter(robot_pose = robot.get_tcp_pose())

    # Initialize keyboard listener
    global loop, state, finish_time, finish_rating
    listener = keyboard.Listener(on_press = _on_press, on_release = _on_release)
    listener.start()

    # Initialize folders
    folder_name = "task_{:04d}_user_{:04d}_scene_{:04d}_cfg_{:04d}".format(cfg.task_id, cfg.user_id, cfg.scene_id, cfg.cfg_id)
    path = os.path.join(cfg.teleop.data_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Call subprocesses to collect data
    logger.info("Initialize collectors ...")
    temp_file_list = []
    for serial in (list(cfg.hardware.camera["global"].keys()) + list(cfg.hardware.camera["inhand"].keys())):
        cfg.collector_serial = serial
        # Save temporary configuration files
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".yaml") as temp_file:
            OmegaConf.save(cfg, temp_file.name)
            temp_file_list.append(temp_file.name)
        # Open subprocess to collect data
        temp_file_path = temp_file_list[-1]
        subprocess.Popen(
            [
                "python",
                "-m",
                "teledata.collector",
                "--config-dir={}".format(temp_file_path.rsplit('/', 1)[0]),
                "--config-name={}".format(temp_file_path.rsplit('/', 1)[1]),
                "hydra/hydra_logging=disabled",
                "hydra/job_logging=disabled"
            ]
        )
    
    time.sleep(cfg.wait_time.collector_start)
    logger.info("Start teleoperation!")
    last_gripper_level = -1

    while loop:
        # # Check sigma status
        # sig, status = oculus_reader.dhdGetStatus()
        # if not oculus_reader.drdIsRunning() or sig != 0 or status[5] == 0:
        #     logger.warning("Sigma error: invalid status! If this warning message continues, please press 'q' to stop data collection.")
        #     continue

        # # Control robot and gripper according to sigma's output
        # sig, px, py, pz, _, _, _, width, pose = pysigma.drdGetPositionAndOrientation()
        # if sig != 0:
        #     logger.warning("Sigma error: invalid return signal! If this warning message continues, please press 'q' to stop data collection.")
        #     continue
        
        # # Get pedal's output, 1x4
        # pedal_ret = pedal.get_axes()
        
        vr_transforms,vr_buttons = oculus_reader.get_transformations_and_buttons()

        # Convert to robot tcp pose and send the pose to the robot
        tcp_pose = converter.step(vr_transforms)
        robot.send_tcp_pose(tcp_pose)
        
        # # Send the gripper width to the gripper
        # try:
        #     grasp_level = int(np.clip(width_coeff * width / cfg.teleop.sigma_max_width, 0, 1) * cfg.teleop.gripper_quantization_steps)
        #     if grasp_level != last_gripper_level:
        #         grasp_width = grasp_level * gripper_quantization_size
        #         gripper.set_width(grasp_width) 
        #         last_gripper_level = grasp_level
        # except Exception as e:
        #     logger.warning("Gripper error: {}".format(e))
        #     pass

    # Process metadata
    camera_info = {}
    for cam_serial in cfg.hardware.camera["inhand"].keys():
        camera_info[cam_serial] = "inhand"
    for cam_serial in cfg.hardware.camera["global"].keys():
        camera_info[cam_serial] = "global"
    all_keys = list(KEYS["robot"][cfg.hardware.type.robot].keys()) + list(KEYS["gripper"][cfg.hardware.type.gripper].keys())
    extra_keys = [key for key in all_keys if key not in ["tcp", "joint", "ee_state", "ee_command"]]
    meta = {
        "finish_time": finish_time,
        "rating": finish_rating,
        "end_effector": "gripper",
        "robot_type": "single_arm",
        "camera_info": camera_info,
        "main_camera": main_serial,
        "extra_keys": extra_keys
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f)
    
    # Clean-ups
    time.sleep(cfg.wait_time.collector_stop)
    for temp_file_path in temp_file_list:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    listener.stop()
    robot.stop()
    gripper.stop()
    main_camera.stop()

if __name__ == '__main__':
    main()