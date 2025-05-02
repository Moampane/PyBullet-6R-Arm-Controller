from robot_controller import RobotController
import numpy as np
import pybullet as p

# Make numpy print nice
np.set_printoptions(precision=3, suppress=True)

PI = 3.14159

controller = RobotController() # Initialize controller
controller.create_world(GUI=True) # Configure simulation

configuration = [0, -PI/2, PI/2, PI, 0, PI]
ee_pose = controller.forward_kinematics(configuration)
print(ee_pose)

controller.set_joint_configuration(configuration)

for i in range(100):
    p.stepSimulation()

controller.solveForwardPositonKinematics()

while True:
    p.stepSimulation()