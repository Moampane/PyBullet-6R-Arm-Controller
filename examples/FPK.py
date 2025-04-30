import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robot_controller import RobotController
import numpy as np

# Make numpy print nice
np.set_printoptions(precision=3, suppress=True)

controller = RobotController() # Initialize controller
controller.create_world(GUI=True) # Configure simulation

# Calculate end effector pose
PI = 3.14159
configuration = [PI, -PI/2, PI/2, PI, 0, PI]
ee_pose = controller.forward_kinematics(configuration)
print(ee_pose)

# Bring arm to configuration in simulation
controller.set_joint_configuration(configuration)

# Start simulation
controller.start_sim()

