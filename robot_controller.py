import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from random import random
from scipy.linalg import logm

PI = 3.14159

class RobotController:
    def __init__(self, robot_type = 'ur5', controllable_joints = None, end_eff_index = None, time_step = 1e-3):
        self.robot_type = robot_type
        self.robot_id = None
        self.num_joints = None
        self.controllable_joints = controllable_joints
        self.end_eff_index = end_eff_index
        self.time_step = time_step
        self.initial_screw_axes = None

    def create_world(self, GUI=True):
        # load pybullet physics engine
        if GUI:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        GRAVITY = -9.8
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10)
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf")

        #loading robot into the environment
        urdf_file = 'urdf/' + self.robot_type + '.urdf'
        self.robot_id = p.loadURDF(urdf_file, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id) # Joints
        print('#Joints:',self.num_joints)
        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, self.num_joints-1))
        print('#Controllable Joints:', self.controllable_joints)
        if self.end_eff_index is None:
            self.end_eff_index = self.controllable_joints[-1]
        print('#End-effector:', self.end_eff_index)
        if self.initial_screw_axes is None:
            self.initial_screw_axes = get_screw_axes(self.robot_id)

    def start_sim(self):
        while True:
                p.stepSimulation()
                time.sleep(self.time_step)

    def set_joint_configuration(self, configuration):
        p.setJointMotorControlArray(bodyUniqueId = self.robot_id,
                        jointIndices = self.controllable_joints,
                        controlMode = p.POSITION_CONTROL,
                        targetPositions = configuration)

    def forward_kinematics(self, configuration):
        screw_axes = get_screw_axes(self.robot_id)
        pose = np.eye(4)

        for idx in range(len(screw_axes)):
            exponential = calculate_exponential(screw_axes[idx], configuration[idx])
            pose = pose @ exponential

        M_mat = np.array([[-1, 0, 0, 0.817], 
                          [0, 1, 0, 0.109],
                          [0, 0, -1, 0.995],
                          [0, 0, 0, 1]])

        return pose @ M_mat
    
    def inverse_kinematics(self, target, ang_error_thresh, lin_error_thresh, configuration):
        # target is a htm representing the pose you want to place your end effector at
        # configuration is your initial guess
        # variable to be updated
        ang_error = 100.0
        lin_error = 100.0
        ee_pose = None
        theta_dot = []
        count = 0

        while ang_error > ang_error_thresh or lin_error > lin_error_thresh:

            if np.sum(np.absolute(theta_dot)) < 2: # guess new configuration if it appears to be at a local minimum, change in configuration is very small
                configuration = [random()*2*PI, random()*2*PI, random()*PI, random()*2*PI, random()*2*PI, random()*2*PI]
                print("RANDOMIZE")

            mat_v_b = logm(np.linalg.inv(self.forward_kinematics(configuration)) @ target).astype(float)
            v_b = twistm_to_vector(mat_v_b)

            jacobian = self.get_jacobian(configuration)
            theta_dot = np.linalg.solve(jacobian, v_b)
            
            new_theta = np.add(configuration, theta_dot.T)
            configuration = new_theta[0]
            configuration = [joint_pos % 2*PI for joint_pos in configuration]

            if abs(configuration[2]) > PI: # 3rd joint limit, must be between -PI and PI
                continue

            ee_pose = self.forward_kinematics(configuration)

            ang_error = get_angular_error(ee_pose, target)
            lin_error = get_translational_error(ee_pose, target)
            count+=1
            print(count)

        print(f"estimated: {ee_pose}")
        print(f"estimated config: {configuration}")

        print(f"actual: {target}")
        return configuration
    
    def get_jacobian(self, configuration):
        exponentials = [calculate_exponential(self.initial_screw_axes[idx], configuration[idx]) for idx in range(len(configuration)-1)]

        jacobian_columns = [self.initial_screw_axes[0]]
        for idx in range(5):
            exp_product = np.eye(4)
            for exponential in exponentials[:idx+1]:
                exp_product = exp_product @ exponential
            adj_transform = get_adj_transform(exp_product)
            jacobian_columns.append(adj_transform @ self.initial_screw_axes[idx+1])

        jacobian = np.column_stack(jacobian_columns)

        return jacobian

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

def htm_to_vector(htm):

    if abs(htm[2][0]) < 1:
        pitch = -np.arcsin(htm[2][0]) # y-axis
        roll = np.arctan2(htm[2][1], htm[2][2]) # x-axis
        yaw = np.arctan2(htm[1][0], htm[0][0]) # z-axis
    else:
        # Gimbal lock
        pitch = PI / 2 if htm[2][0] <= -1 else -PI / 2
        roll = 0
        yaw = np.arctan2(-htm[0][1], htm[1])

    x = htm[0][3]
    y = htm[1][3]
    z = htm[2][3]
    
    vector = np.array([
        [roll],
        [pitch],
        [yaw],
        [x],
        [y],
        [z]
    ])
    return vector

def twistm_to_vector(htm):

    roll = htm[2][1]
    pitch = htm[0][2]
    yaw = htm[1][0]
    x = htm[0][3]
    y = htm[1][3]
    z = htm[2][3]
    
    vector = np.array([
        [roll],
        [pitch],
        [yaw],
        [x],
        [y],
        [z]
    ])
    return vector

def get_adj_transform(exponential):
    rot = exponential[:3, :3]
    lin = np.array(exponential[3, :3])
    adj_transformation = np.concatenate((np.concatenate((rot, np.zeros((3, 3))), 1),
                        np.concatenate((skew_mat(lin) @ rot, rot), 1)))
    
    return adj_transformation

def get_angular_error(htm1, htm2):
    # Calculate error
    vec1 = htm_to_vector(htm1)
    vec2 = htm_to_vector(htm2)

    # Angular error
    ang_vec1 = vec1[:3].T
    ang_vec2 = vec2[:3].T

    r1 = R.from_euler('zyx', ang_vec1, degrees = False)
    r2 = R.from_euler('zyx', ang_vec2, degrees = False)

    # Relative rotation matrix
    relative_rotation = r1.inv() * r2

    # Compute angular difference (in radians)
    error_rad = relative_rotation.magnitude()

    return error_rad

def get_translational_error(htm1, htm2):
    # Calculate linear error
    vec1 = htm_to_vector(htm1)
    vec2 = htm_to_vector(htm2)

    lin1 = vec1[3:]
    lin2 = vec2[3:]

    error_lin = np.sqrt((lin1[0]-lin2[0])**2 + (lin1[1]-lin2[1])**2 + (lin1[2]-lin2[2])**2)

    return error_lin