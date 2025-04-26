import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
GRAVITY = -9.8
TIME_STEP = 1./240.
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

while True:
    p.stepSimulation()
    time.sleep(TIME_STEP)
