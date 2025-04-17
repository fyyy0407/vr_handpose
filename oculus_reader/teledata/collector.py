"""
Main Data Collection Script.
"""


"""
Teleoperation Script.
"""

import os
import cv2
import time
import json
import hydra
import logging
import numpy as np

from pynput import keyboard
from omegaconf import OmegaConf
from easyrobot.utils.logger import ColoredLogger
from easyrobot.utils.shared_memory import SharedMemoryManager

from teledata.constants import *
from teledata.utils import calc_calib, resolve_logger


loop = True

def _on_press(key):
    global loop
    try:
        if key.char == 'q':
            loop = False
            return False
    except AttributeError:
        pass
    
def _on_release(key):
    pass


@hydra.main(
    version_base = None,
    config_path = None
)
def main(cfg):
    OmegaConf.resolve(cfg)
    
    # Find the serial
    serial = cfg.collector_serial

    # Setup logger
    logger = resolve_logger("{} {}".format(cfg.logger.collector, serial))

    folder_name = "task_{:04d}_user_{:04d}_scene_{:04d}_cfg_{:04d}".format(cfg.task_id, cfg.user_id, cfg.scene_id, cfg.cfg_id)
    path = os.path.join(cfg.teleop.data_path, folder_name)
    main_serial = cfg.hardware.camera.main
    
    assert serial in (list(cfg.hardware.camera["global"].keys()) + list(cfg.hardware.camera["inhand"].keys())), \
        "Cannot find camera {} in camera list.".format(serial)
    is_main = (serial == main_serial)

    if is_main:
        target_duration = 0 if cfg.collector_freq.main is None else 1.0 / cfg.collector_freq.main
    else:
        target_duration = 0 if cfg.collector_freq.others is None else 1.0 / cfg.collector_freq.others

    # Initialize camera folders
    camera_path = os.path.join(path, "cam_{}".format(serial))
    if not os.path.exists(camera_path):
        os.makedirs(camera_path)
    color_path = os.path.join(camera_path, "color")
    if not os.path.exists(color_path):
        os.makedirs(color_path)
    depth_path = os.path.join(camera_path, "depth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    if is_main:
        # Initialize lowdim folders
        lowdim_path = os.path.join(path, "lowdim")
        if not os.path.exists(lowdim_path):
            os.makedirs(lowdim_path)

        # Initialize camera shared memory
        main_resolution = cfg.hardware.camera["global" if serial in cfg.hardware.camera["global"].keys() else "inhand"][serial].resolution
        shm_camera_color = SharedMemoryManager(cfg.shm_name.main_camera_color, 1, (main_resolution[1], main_resolution[0], 3), np.uint8)
        shm_camera_depth = SharedMemoryManager(cfg.shm_name.main_camera_depth, 1, (main_resolution[1], main_resolution[0]), np.uint16)
        
        # Initialize lowdim shared memory
        shm_robot = SharedMemoryManager(cfg.shm_name.robot, 1, tuple(SHAPE["robot"][cfg.hardware.type.robot]), np.float32)
        
        shm_rohand = SharedMemoryManager(cfg.shm_name.rohand, 1, tuple(SHAPE["rohand"]), np.float32)

        # Initialize lowdim memory
        lowdim_memory = {}
        for key in KEYS["robot"][cfg.hardware.type.robot].keys():
            lowdim_memory[key] = {}
        for key in KEYS["rohand"].keys():
            lowdim_memory[key] = {}
    else:
        logger.info("Initialize camera.")
        # Initialize camera
        camera = hydra.utils.instantiate(
            cfg.hardware.camera["global" if serial in cfg.hardware.camera["global"].keys() else "inhand"][serial], 
            serial = serial
        )

    # Process camera extrinsics
    intrinsic, extrinsic = calc_calib(cfg.calib.data_path, cfg.cfg_id, serial, (serial in cfg.hardware.camera["global"].keys()))
    np.save(os.path.join(camera_path, "intrinsic.npy"), intrinsic)
    with open(os.path.join(camera_path, "extrinsic.json"), "w") as f:
        json.dump(extrinsic, f)
    
    # Initialize keyboard listener
    global loop
    listener = keyboard.Listener(on_press = _on_press, on_release = _on_release)
    listener.start()

    logger.info("Collector ready.")

    while loop:
        tic = time.time()

        # Get observations
        timestamp = int(time.time() * 1000)
        if is_main:
            color_res = shm_camera_color.execute()
            depth_res = shm_camera_depth.execute()
            robot_res = shm_robot.execute()
            rohand_res = shm_rohand.execute()
        else:
            color_res, depth_res = camera.get_rgbd_image()

        # Save images
        cv2.imwrite(os.path.join(color_path, '{}.png'.format(timestamp)), color_res[:, :, ::-1])
        cv2.imwrite(os.path.join(depth_path, '{}.png'.format(timestamp)), depth_res)

        # Process observations
        if is_main:
            for key, func in KEYS["robot"][cfg.hardware.robot].items():
                lowdim_memory[key][str(timestamp)] = func(robot_res)
            for key, func in KEYS["rohand"].items():
                lowdim_memory[key][str(timestamp)] = func(rohand_res)
        
        # Ensure frequency
        duration = time.time() - tic
        if duration < target_duration:
            time.sleep(target_duration - duration)
    
    listener.stop()

    logger.info("Collector stop.")
    
    if is_main:
        logger.info("Saving lowdim files ...")
        for key, memory in lowdim_memory.items():
            np.savez(os.path.join(lowdim_path, "{}.npz".format(key)), **memory)        
        shm_robot.close()
        shm_gripper.close()
        shm_camera_color.close()
        shm_camera_depth.close()
    else:
        camera.stop()
    
    logger.info("Collector close.")
    

if __name__ == '__main__':
    main()