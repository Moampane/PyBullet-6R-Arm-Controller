import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robot_controller import RobotController

controller = RobotController() # Initialize controller
controller.create_world(GUI=True) # Configure simulation

controller.do_free_fall() # let arm fall

# Start simulation
controller.start_sim()