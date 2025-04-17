import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

class PoseConverter:
    def __init__(
        self,
        robot_pose
    ):
        super(PoseConverter, self).__init__()
        self.init_robot_rot = quat2mat(robot_pose[3:])
        self.robot_pos = robot_pose[:3]
        self.robot_rot = self.init_robot_rot
        self.vr_pos = None

    def step(self, vr_matrix):
        vr_pos = vr_matrix[:3,3]
        vr_rot = vr_matrix[:3,:3]
        if self.vr_pos is not None:
            vr2rob = np.array([[0,0,1],[1,0,0],[0,1,0]])
            translation = vr_pos - self.vr_pos
            translation = vr2rob @ translation
            print("trans: ",translation)
            self.robot_pos = self.robot_pos +  translation
            
            self.robot_rot =vr2rob @ vr_rot @ self.init_robot_rot
        self.vr_pos = vr_pos
        robot_pose = np.concatenate([self.robot_pos, mat2quat(self.robot_rot)], axis = -1)
        return robot_pose
