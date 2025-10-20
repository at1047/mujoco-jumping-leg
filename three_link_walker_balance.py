import mujoco
import mujoco.viewer
import mujoco_py
import numpy as np
import time


# Load your model
model = mujoco.MjModel.from_xml_path("three_link_walker.xml")
data = mujoco.MjData(model)

# Initial state: [x, y, q1, q2, q3, xdot, ydot, q1dot, q2dot, q3dot]
x0 = np.array([0, 0.2970, 0.2, 0.2, 0.0,   # initial positions
               0, 0, 0, 0, 0])              # initial velocities

# Set initial positions and velocities
# Based on the XML structure: pin, pin2, pin3 are the joint names
# Map x0 to MuJoCo coordinates: [x, y, q1, q2, q3, xdot, ydot, q1dot, q2dot, q3dot]
# Note: x, y are not directly used in this model as it's a pendulum system
# The joints are: pin (q1), pin2 (q2), pin3 (q3)

# Set joint positions (q1, q2, q3)
data.qpos[2] = x0[4]  # q1 (pin joint)
data.qpos[3] = x0[3]  # q2 (pin2 joint) 
data.qpos[4] = -x0[2]  # q3 (pin3 joint)

# Set joint velocities (q1dot, q2dot, q3dot)
data.qvel[2] = x0[9]  # q1dot
data.qvel[3] = x0[8]  # q2dot
data.qvel[4] = -x0[7]  # q3dot

# Forward kinematics to update the model state
mujoco.mj_forward(model, data)

# def controller(model, data):
#     """Simple controller - can be modified as needed"""
#     # For now, just set control inputs to zero (no control)
#     data.ctrl[:] = 0.0


with mujoco.viewer.launch_passive(model, data) as viewer:
    # Option 1: Set camera properties manually
    viewer.cam.distance = 7.0      # Distance from target
    viewer.cam.azimuth = 45.0      # Horizontal angle (0-360 degrees)
    viewer.cam.elevation = -30.0   # Vertical angle (-90 to 90 degrees)
    viewer.cam.lookat[:] = [0, 0, 3]  # Point to look at [x, y, z]
    
    # Option 2: Use predefined cameras from XML
    # viewer.cam.fixedcamid = model.camera("side_view").id  # Use side_view camera
    # viewer.cam.fixedcamid = model.camera("top_view").id   # Use top_view camera
    
    print("Starting simulation...")
    step_count = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Print status every 100 steps
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step {step_count}: qpos = {data.qpos}, qvel = {data.qvel}")
        
        # Use model timestep for proper timing
        time.sleep(model.opt.timestep)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # Set camera properties for the second viewer
#     viewer.cam.distance = 8.0      # Distance from target
#     viewer.cam.azimuth = 30.0      # Horizontal angle (0-360 degrees)
#     viewer.cam.elevation = -30.0   # Vertical angle (-90 to 90 degrees)
#     viewer.cam.lookat[:] = [0, 0, 0.5]  # Point to look at [x, y, z]
    
#     start_time = time.time()
#     while viewer.is_running():
#         current_time = time.time() - start_time
#         dt = model.opt.timestep

        # q = data.qpos.copy()
        # qdot = data.qvel.copy()

        # q_theta = T @ q.T + d
        # dq_theta = T @ qdot.T

        # q3 = q_theta[2]
        # q4 = q_theta[3]
        # q5 = q_theta[4]

        # y = np.array([[q3 - 0.3],
        #               [q4 + q5]])

        # y1 = y[0]
        # y2 = y[1]

        # dq3 = dq_theta[2]
        # dq4 = dq_theta[3]
        # dq5 = dq_theta[4]

        # ydot = np.array([[dq3],
        #                   [q5]])

        # ydot1 = ydot[0]
        # ydot2 = ydot[1]

        # eps = 0.1
        # a = 0.9

        # M = np.zeros((5,5))
        # mujoco.mj_fullM(model, M, data.qM)
        # Cqdot = data.qfrc_bias
        # 
        # # desired acceleration from psi_a law
        # v1 = (1/eps**2) * psi_a(y1, eps*ydot1, a)
        # v2 = (1/eps**2) * psi_a(y2, eps*ydot2, a)
        # v = np.array([v1, v2])

        # Kp = 500
        # Kd = 100
        # 
        # # feedback torque
        # tau = -Kp * np.array([y1, y2]) - Kd * np.array([ydot1, ydot2]) + v
        # print(tau)

        # data.ctrl[:] = tau.T

        # controller(model, data) 

        # mujoco.mj_step(model, data)

        # viewer.sync()
        # time.sleep(0.01)
        # time.sleep(dt)

