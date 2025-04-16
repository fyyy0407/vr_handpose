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
from teledata.converter import PoseConverter


from OyMotion.Glove import *
from OyMotion.ROHand import *

loop = True
state = 0
finish_time = 0
finish_rating = 0

def configure_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 如果logger已有handler则不重复添加
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

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


def recorder_process_glove(queue, folder_path):
    """
    子进程函数：不断从队列中读取灵巧手数据（各手指位置），写入 JSONL 文件
    """
    glove_obs_file = os.path.join(folder_path, "glove_observations.jsonl")
    with open(glove_obs_file, "w") as f:
        while True:
            glove_obs = queue.get()
            if glove_obs is None:
                break
            f.write(json.dumps(glove_obs) + "\n")
            f.flush()


async def glove_control_main(glove_queue,NODE_ID,COM_PORT,calib):
    """
    Finger Control
    """
    client = ROHand(COM_PORT, NODE_ID)
    client.connect()
    glove_ctrl = Glove()
    await glove_ctrl.connect_gforce_device()

    await glove_ctrl.calib(calib)
    print("finish calib")
    client.reset()
    print("finish reset")

    try:
        while not glove_ctrl.terminated:
            await glove_ctrl.get_pos()
            # 直接调用 set_finger_pos 控制灵巧手的每个手指
            # resp = client.set_finger_pos(ROH_FINGER_POS_TARGET0, glove_ctrl.finger_data)
            print("finger pose: ",glove_ctrl.finger_data)
            # record data
            finger_positions = client.get_current_pos()
            glove_data = {
                "timestamp": int(time.time() * 1000),
                "finger_positions": finger_positions  # 例如一个列表或字典
            }
            glove_queue.put(glove_data)
            await asyncio.sleep(0.05)
    finally:
        await glove_ctrl.gforce_device.stop_streaming()
        await glove_ctrl.gforce_device.disconnect()

def start_glove_control(glove_queue,NODE_ID,COM_PORT,calib):
    asyncio.run(glove_control_main(glove_queue,NODE_ID,COM_PORT,calib))
    
@hydra.main(
    version_base=None,
    config_path=os.path.join("configs"),
    config_name="config"
)
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    # Setup logger
    logger = configure_logger("teleop", level=logging.INFO)

    ############## INITIALIZATION #################
    logger.info("Initialization......")
    robot = hydra.utils.instantiate(
        cfg.hardware.robot, 
        # shm_name=cfg.shm_name.robot
    )    
    robot_init_pose = np.array(cfg.pose.init.robot)
    # robot.send_tcp_pose(robot_init_pose)
    print("init pose: ",robot_init_pose)
    
    # Initialize camera, assuming that the first global camera is the main camera.
    # main_serial = cfg.hardware.camera.main
    # logger.info("Initialize main camera {} ...".format(main_serial))
    # assert main_serial in (list(cfg.hardware.camera["global"].keys()) + list(cfg.hardware.camera["inhand"].keys())), \
    #     "Cannot find main camera {} in camera list.".format(main_serial)
    # main_camera = hydra.utils.instantiate(
    #     cfg.hardware.camera["global" if main_serial in cfg.hardware.camera["global"].keys() else "inhand"][main_serial], 
    #     serial = main_serial,
    #     shm_name_rgb = cfg.shm_name.main_camera_color,
    #     shm_name_depth = cfg.shm_name.main_camera_depth
    # )
    oculus_reader = hydra.utils.instantiate(
        cfg.hardware.oculus_reader,
        # shm_name=cfg.shm_name.oculus_reader
    )
    
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

    # 创建用于跨进程传递数据的队列，并启动子进程记录观测数据
    data_queue = multiprocessing.Queue()

    
    # # 创建新的队列用于记录灵巧手数据，并启动对应的子进程
    # glove_queue = multiprocessing.Queue()
    # glove_recorder_proc = multiprocessing.Process(target=recorder_process_glove, args=(glove_queue, path))
    # glove_recorder_proc.start()

    # 启动手套控制线程，用于控制灵巧手
    # node_id = cfg.hardware.glove_control.node_id
    # com_port = cfg.hardware.glove_control.com_port
    # calib = cfg.hardware.glove_control.calib
    # glove_thread = threading.Thread(target=start_glove_control, args=(glove_queue,node_id,com_port,calib),daemon=True)
    # glove_thread.start()

    
    # # Call subprocesses to collect data
    # logger.info("Initialize collectors ...")
    # temp_file_list = []
    # for serial in (list(cfg.hardware.camera["global"].keys()) + list(cfg.hardware.camera["inhand"].keys())):
    #     cfg.collector_serial = serial
    #     # Save temporary configuration files
    #     with tempfile.NamedTemporaryFile(delete = False, suffix = ".yaml") as temp_file:
    #         OmegaConf.save(cfg, temp_file.name)
    #         temp_file_list.append(temp_file.name)
    #     # Open subprocess to collect data
    #     temp_file_path = temp_file_list[-1]
    #     subprocess.Popen(
    #         [
    #             "python",
    #             "-m",
    #             "teledata.collector",
    #             "--config-dir={}".format(temp_file_path.rsplit('/', 1)[0]),
    #             "--config-name={}".format(temp_file_path.rsplit('/', 1)[1]),
    #             "hydra/hydra_logging=disabled",
    #             "hydra/job_logging=disabled"
    #         ]
    #     )
    
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
         
        ################ Record Operation and Observation ################

        timestamp = int(time.time() * 1000)
        
        # 将所有数据打包成字典，注意将 tcp_pose 转换为列表
        # obs = {
        #     "timestamp": timestamp,
        #     "tcp_pose": tcp_pose.tolist(),
        # }
        
        # 将观测数据发送给记录子进程（异步写入）
        # data_queue.put(obs)
        
        time.sleep(0.05)  

    # camera_info = {}
    # for cam_serial in cfg.hardware.camera["inhand"].keys():
    #     camera_info[cam_serial] = "inhand"
    # for cam_serial in cfg.hardware.camera["global"].keys():
    #     camera_info[cam_serial] = "global"
    # all_keys = list(KEYS["robot"][cfg.hardware.type.robot].keys()) + list(KEYS["gripper"][cfg.hardware.type.gripper].keys())
    # extra_keys = [key for key in all_keys if key not in ["tcp", "joint", "ee_state", "ee_command"]]
    # meta = {
    #     "finish_time": finish_time,
    #     "rating": finish_rating,
    #     "robot_type": "single_arm",
    #     "camera_info": camera_info,
    #     "main_camera": main_serial,
    #     "extra_keys": extra_keys
    # }
    # with open(os.path.join(path, "metadata.json"), "w") as f:
    #     json.dump(meta, f)
    
    data_queue.put(None)
    # glove_queue.put(None)
    listener.stop()
    # main_camera.stop()
    robot.stop() 
    # glove_recorder_proc.join()
    # glove_thread.join(timeout=1)

if __name__ == '__main__':
    main()