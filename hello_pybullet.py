import pybullet as p
from helpers import make_htm, forward_kinematics
import time
import pybullet_data
import numpy as np

# Make numpy print nice
np.set_printoptions(precision=3, suppress=True)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
GRAVITY = -9.8
TIME_STEP = 1./240.
PI = 3.14159
p.setGravity(0,0,GRAVITY)
p.setTimeStep(TIME_STEP)
p.setPhysicsEngineParameter(fixedTimeStep=TIME_STEP, numSolverIterations=100, numSubSteps=10)
p.setRealTimeSimulation(True)
p.loadURDF("plane.urdf")

#loading robot into the environment
robot_id = p.loadURDF('/urdf/ur5.urdf', useFixedBase=True)

num_joints = p.getNumJoints(robot_id) # Joints
print('#Joints:', num_joints)
controllable_joints = list(range(1, num_joints-1))
print('#Controllable Joints:', controllable_joints)

# Choose configuration
configuration = [PI, -PI/2, PI/2, PI, 0, PI]

print("Calculated")
pose = forward_kinematics(robot_id, configuration)
print(pose)

# Validate FK
p.setJointMotorControlArray(bodyUniqueId = robot_id,
                        jointIndices = controllable_joints,
                        controlMode = p.POSITION_CONTROL,
                        targetPositions = configuration)
# Get robot in position to calculate actual
for i in range(100):
    p.stepSimulation()

eeState = p.getLinkState(bodyUniqueId = robot_id, linkIndex = controllable_joints[-1])
eePose = make_htm(eeState[0], eeState[1])
print("Actual")
print(eePose)

while True:
    p.stepSimulation()
    time.sleep(TIME_STEP)
    # joint_info = p.getJointStates(bodyUniqueId = robot_id, jointIndices = controllable_joints)
    # print("----------------")
    # for info in joint_info:
    #     print(info)
