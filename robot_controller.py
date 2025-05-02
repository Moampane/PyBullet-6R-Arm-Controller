import pybullet as p
import pybullet_data
import numpy as np
import time

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

    pitch = np.arcsin(-htm[2][0]) # y-axis
    roll = np.arctan2(htm[2][1], htm[2][2]) # x-axis
    yaw = np.arctan2(htm[1][0], htm[0][0]) # z-axis

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