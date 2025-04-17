'''
RealSense Camera.
'''

import numpy as np
import pyrealsense2 as rs
from multiprocessing import shared_memory


class RealSenseRGBDCamera:
    '''
    RealSense RGB-D Camera.
    '''
    def __init__(
        self, 
        serial, 
        frame_rate = 30, 
        resolution = (1280, 720),
        align = True,
        shm_name_rgb = None,
        shm_name_depth = None,
        **kwargs
    ):
        '''
        Initialization.

        Parameters:
        - serial: str, required, the serial number of the realsense device;
        - frame_rate: int, optional, default: 15, the framerate of the realsense camera;
        - resolution: (int, int), optional, default: (1280, 720), the resolution of the realsense camera;
        - align: bool, optional, default: True, whether align the frameset with the RGB image.
        '''
        super(RealSenseRGBDCamera, self).__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        # =============== Support L515 Camera ============== #
        self.is_radar = str.isalpha(serial[0])
        depth_resolution = (1024, 768) if self.is_radar else resolution
        if self.is_radar:
            frame_rate = max(frame_rate, 30)
            self.depth_scale = 4000
        else:
            self.depth_scale = 1000
        # ================================================== #
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)
        self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.with_align = align
        
        # shared memory
        H, W = resolution[1], resolution[0]
        self.rgb_shape   = (H, W, 3)
        self.depth_shape = (H, W)
        
         # 创建或打开共享内存
        if shm_name_rgb and shm_name_depth:
            # 主进程：create=True；其它进程：create=False
            self.shm_rgb   = shared_memory.SharedMemory(name=shm_name_rgb,   create=True,
                                    size=np.prod(self.rgb_shape)  * np.dtype(np.uint8).itemsize)
            self.shm_depth = shared_memory.SharedMemory(name=shm_name_depth, create=True,
                                    size=np.prod(self.depth_shape)* np.dtype(np.uint16).itemsize)
            # 映射成 NumPy 数组视图
            self.buf_rgb   = np.ndarray(self.rgb_shape,   dtype=np.uint8,   buffer=self.shm_rgb.buf)
            self.buf_depth = np.ndarray(self.depth_shape, dtype=np.uint16, buffer=self.shm_depth.buf)
        else:
            self.shm_depth = self.shm_rgb = None

    def get_rgb_image(self):
        '''
        Get the RGB image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        if self.shm_rgb:
            self.buf_rgb[:] = color_image
        return color_image

    def get_depth_image(self):
        '''
        Get the depth image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.uint16) 
        if self.shm_depth:
            self.buf_depth[:] = depth_image
        return depth_image / self.depth_scale

    def get_rgbd_image(self):
        '''
        Get the RGB image along with the depth image from the camera.
        '''
        frameset = self.pipeline.wait_for_frames()
        if self.with_align:
            frameset = self.align.process(frameset)
        color_image = np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8)
        depth_image = np.asanyarray(frameset.get_depth_frame().get_data()).astype(np.uint16)
        if self.shm_rgb and self.shm_depth:
            self.buf_rgb[:]   = color_image
            self.buf_depth[:] = depth_image
        return color_image, depth_image / self.depth_scale
    
    def get_info(self):
        color, depth = self.get_rgbd_image()
        return color, depth
    
    def stop(self):
        self.pipeline.stop()
        self.shm_rgb.close();   self.shm_depth.close()
        self.shm_rgb.unlink();  self.shm_depth.unlink()