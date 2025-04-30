import numpy as np
import pybullet as p


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def make_htm(position, orientation):
    orientation = list(p.getMatrixFromQuaternion(orientation))
    htm = np.array([orientation[:3]+[position[0]],
               orientation[3:6]+[position[1]],
               orientation[6:]+[position[2]],
               [0, 0, 0, 1]])
    return htm

def get_M_mat(robot_id):
    num_joints = p.getNumJoints(robot_id)
    controllable_joints = list(range(1, num_joints-1))
    eeState = p.getLinkState(bodyUniqueId = robot_id, linkIndex = controllable_joints[-1])
    M_mat = make_htm(eeState[0], eeState[1])

    return M_mat

def get_screw_axes(robot_id):
    num_joints = p.getNumJoints(robot_id)
    controllable_joints = list(range(1, num_joints-1))
    screw_axes = []
    joint_positions_world = [[0, 0, 1.089159], [0, 0.13585, 1.089159], [0.425, 0.01615, 1.089159], [0.81725, 0.01615, 1.089159], [0.81725, 0.10915, 1.089159], [0.81725, 0.10915, 0.994509]]

    for joint_idx in controllable_joints:
        info = p.getJointInfo(robot_id, joint_idx)

        # Position of joint in world frame
        joint_pos_world = np.array(joint_positions_world[joint_idx-1])  

        # Joint axis in local (link) frame
        joint_axis_local = np.array(info[13])

        # Get world transform of parent link to rotate axis into world frame
        parent_index = info[16]
        if parent_index == -1:
            parent_world_rot = np.eye(3)
        else:
            parent_state = p.getLinkState(robot_id, parent_index, computeForwardKinematics=True)
            parent_world_quat = parent_state[1]
            parent_world_rot = np.array(p.getMatrixFromQuaternion(parent_world_quat)).reshape(3, 3)

        # Rotate axis to world frame
        omega = parent_world_rot @ joint_axis_local

        q = joint_pos_world
        v = -np.cross(omega, q)

        screw_axis = np.concatenate([omega, v])
        screw_axes.append(screw_axis)

    return screw_axes

def skew_mat(vector):
    return np.array([[0, -vector[2], vector[1]],
                      [vector[2], 0, -vector[0]],
                      [-vector[1], vector[0], 0]])

def angular_exponential(screw_axis, theta):
    s_omega = screw_axis[:3]
    screw_skew_mat = skew_mat(s_omega)
    return np.identity(3) + np.sin(theta)*screw_skew_mat + (1-np.cos(theta))*screw_skew_mat@screw_skew_mat

def calculate_exponential(screw_axis, theta):
    s_v = screw_axis[3:]
    s_omega = screw_axis[:3]
    screw_skew_mat = skew_mat(s_omega)

    angular_component = angular_exponential(screw_axis, theta)
    linear_component = (np.identity(3)*theta + (1-np.cos(theta))*screw_skew_mat + (theta-np.sin(theta))*screw_skew_mat@screw_skew_mat)@s_v
    linear_component = np.transpose(np.array([linear_component]))
    linear_and_angular = np.concatenate((angular_component, linear_component), 1)
    bottom_row = np.array([[0, 0, 0, 1]])

    return np.concatenate((linear_and_angular, bottom_row))

def forward_kinematics(robot_id, configuration):
    screw_axes = get_screw_axes(robot_id=robot_id)
    pose = np.eye(4)

    for idx in range(len(screw_axes)):
        exponential = calculate_exponential(screw_axes[idx], configuration[idx])
        pose = pose @ exponential

    M_mat = get_M_mat(robot_id)

    return pose @ M_mat