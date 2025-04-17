import os
import cv2
import json
import time
import hydra
import logging
import tempfile
import subprocess
import numpy as np
import asyncio
import threading
import multiprocessing
from omegaconf import OmegaConf

from pynput import keyboard


from teledata.constants import *
from teledata.utils import resolve_logger
from teledata.converter import PoseConverter


from device.OyMotion.Glove import Glove
from device.OyMotion.ROHand import *
from device.oculus_reader.reader import OculusReader

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
    version_base=None,
    config_path=os.path.join("configs"),
    config_name="config"
)
def main(cfg):
    # Setup logger
    OmegaConf.resolve(cfg)
    logger = resolve_logger(cfg.logger.teleop)
    
    ############## INITIALIZATION #################
    logger.info("Initialize teleop device")
    glove_ctrl = Glove()
    await glove_ctrl.connect_gforce_device()
    await glove_ctrl.calib(cfg.hardware.glove.calib)
    
    logger.info("Initialize robot and rohand")
    robot = hydra.utils.instantiate(
        cfg.hardware.robot, 
        shm_name=cfg.shm_name.robot
    )    
    robot_init_pose = np.array(cfg.pose.init.robot)
    # robot.send_tcp_pose(robot_init_pose)
    print("init pose: ",robot_init_pose)
    
    rohand = hydra.utils.instantiate(
        cfg.hardware.rohand,
        shm_name=cfg.shm_name.rohand
    )
    rohand.connect()
    rohand.reset()
    
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
    
    logger.info("Initialize oculus reader")
    oculus_reader = OculusReader()
    
    logger.info("Initialize converter")
    converter = PoseConverter(robot_pose=robot.get_tcp_pose())
    
    # Initialize keyboard listener
    global loop, state, finish_time, finish_rating
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    # Initialize folders
    folder_name = "task_{:04d}_user_{:04d}_scene_{:04d}_cfg_{:04d}".format(
        cfg.task_id, cfg.user_id, cfg.scene_id, cfg.cfg_id
    )
    path = os.path.join(cfg.teleop.data_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    logger.info("Initialization Finished.")


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
    
    # time.sleep(cfg.wait_time.collector_start)
    logger.info("Start teleoperation!")

    # 主循环中采集数据，并将数据发送给记录子进程
    while loop:
        time.sleep(0.3)
        vr_transforms, vr_buttons = oculus_reader.get_transformations_and_buttons()
        print('vr_transforms:',vr_transforms)
        tcp_pose = converter.step(vr_transforms['r'])
        # robot.send_tcp_pose(tcp_pose)
        print("new tcp pose: ",tcp_pose)
        
        await glove_ctrl.get_pos()
        resp = rohand.set_finger_pos(ROH_FINGER_POS_TARGET0, glove_ctrl.finger_data)

        
        time.sleep(0.05)  

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
        "robot_type": "single_arm",
        "camera_info": camera_info,
        "main_camera": main_serial,
        "extra_keys": extra_keys
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f)
    

    listener.stop()
    main_camera.stop()
    robot.stop() 
    await glove_ctrl.gforce_device.stop_streaming()
    await glove_ctrl.gforce_device.disconnect()
    


if __name__ == '__main__':
    main()