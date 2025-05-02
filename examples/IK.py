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
configuration = [5*PI/4, -PI/4, PI/2, PI/2, PI, 0] 
linear_error_threshold = 0.1
angular_error_threshold = 0.1
desired_pose = np.array([[1, 0, 0, -0.392],
                [0, -1, 0, -0.109],
                [0, 0, -1, 1.609],
                [0, 0, 0, 1]])

found_configuration = controller.inverse_kinematics(desired_pose, angular_error_threshold, linear_error_threshold, configuration)

# Bring arm to configuration in simulation
controller.set_joint_configuration(found_configuration)

# Start simulation
controller.start_sim()

