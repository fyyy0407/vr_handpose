import numpy as np

# Constants
SHAPE = {
    "robot": {
        "flexiv": [39],
        "realman": [19]
    },
    "gripper": {
        "dahuan": [2],
        "robotiq": [2]
    }
}

KEYS = {
    "robot": {
        "flexiv": {
            "tcp": lambda res: res[0:7],
            "joint": lambda res: res[7:14],
            "tcp_vel": lambda res: res[14:20],
            "joint_vel": lambda res: res[20:27], 
            "force_torque_tcp": lambda res: res[27:33], 
            "force_torque_base": lambda res: res[33:39]
        },
        "realman": {
            "tcp": lambda res: res[0:7],
            "joint": lambda res: res[7:13],
            "force_torque": lambda res: res[13:19]
        }
    },
    "gripper": {
        "dahuan": {
            "ee_state": lambda res: res[0:1],
            "ee_command": lambda res: res[1:2]
        },
        "robotiq": {
            "ee_state": lambda res: res[0:1],
            "ee_command": lambda res: res[1:2]
        }
    }
}

INHAND_TCP_TRANSFORMATION = {
    1: np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0.077],
        [0, 0, 1, 0.2665],
        [0, 0, 0, 1]
    ]),
    8: np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0.077],
        [0, 0, 1, 0.1865],
        [0, 0, 0, 1]
    ]),
    9: np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0.066],
        [0, 0, 1, 0.1973],
        [0, 0, 0, 1]
    ])
}

INHAND_SERIAL = {
    1: "043322070878",
    8: "104422070044",
    9: "819112070044"
}