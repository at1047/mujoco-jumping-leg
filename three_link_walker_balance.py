import mujoco
import mujoco.viewer
import mujoco_py
import numpy as np
import time
import sympy as sp
import util
import symbolic as sym
import states
from numpy import cos, sin

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
print(data.qpos)

# Set joint positions (q1, q2, q3)
data.qpos[0] = x0[0]  # x (torso_x joint)
data.qpos[1] = x0[1]  # y (torso_z joint)

data.qpos[2] = x0[2]  # q1 (pin joint)
data.qpos[3] = -x0[3]  # q2 (pin2 joint) 
data.qpos[4] = x0[3] + x0[4]  # q3 (pin3 joint)

# Set joint velocities (q1dot, q2dot, q3dot)
data.qvel[2] = x0[7]  # q1dot
data.qvel[3] = -x0[8]  # q2dot
data.qvel[4] = x0[8] + x0[9]  # q3dot

# Forward kinematics to update the model state
mujoco.mj_forward(model, data)

# # Print body information including lengths
# for i in range(model.nbody):
#     body_name = model.body(i).name
#     body_pos = model.body(i).pos
#     print(f"Body {i} ({body_name}): pos = {body_pos}")

# # Print geom information (which contains size/length information)
# print(f"Number of geoms: {model.ngeom}")
# for i in range(model.ngeom):
#     geom_name = model.geom(i).name if model.geom(i).name else f"geom_{i}"
#     geom_size = model.geom(i).size
#     geom_type = model.geom(i).type
#     print(f"Geom {i} ({geom_name}): type = {geom_type}, size = {geom_size}")

print(f"Initial qpos: {data.qpos}")
print(f"Initial qvel: {data.qvel}")

# Get and print link lengths
link_lengths = util.get_link_lengths(model)

def compute_u(q1, q2, q1dot, q2dot, y, ydot):
    num1 = (
        3 * (
            369600*np.cos(q1)
            - 1731840*np.cos(q1 - q2)
            - 4477320*np.cos(q1 - q2)**2 * np.sin(q1)**2
            + 5010840*np.sin(q1 - q2)**2 * np.cos(q1)**2
            + 554400*np.cos(q1)*np.cos(q2)
            - 255960*np.sin(q1 - q2)**2 * np.sin(q1)**2
            + 56700*np.cos(q1)**2
            - 246780*np.cos(q1 - q2)*np.cos(q1)
            - 649440*np.cos(q1 - q2)*np.cos(q2)
            - 220500*np.sin(q1)**2
            + 774900*np.sin(q1 - q2)*np.sin(q1)
            - 88560*np.cos(q1 - q2)**2
            + 7350*q1dot**2*np.cos(q1)**2
            + 85050*np.cos(q1)**2*np.cos(q2)
            + 1210320*np.sin(q1 - q2)**2
            + 7350*q1dot**2*np.sin(q1)**2
            - 330750*np.cos(q2)*np.sin(q1)**2
            - 40344*q1dot**2*np.cos(q1 - q2)**2
            - 40344*q2dot**2*np.cos(q1 - q2)**2
            + 285120*np.cos(q1 - q2)*np.cos(q1)**2
            + 95040*np.cos(q1 - q2)**2*np.cos(q1)
            + 43740*np.cos(q1 - q2)*np.cos(q1)**3
            - 33210*np.cos(q1 - q2)**2*np.cos(q2)
            + 4860*np.cos(q1 - q2)**3*np.cos(q1)
            - 40344*q1dot**2*np.sin(q1 - q2)**2
            - 40344*q2dot**2*np.sin(q1 - q2)**2
            - 15482625*np.cos(q1 - q2)*np.sin(q1)**2
            + 18565827*np.sin(q1 - q2)**2*np.cos(q1)
            + 453870*np.sin(q1 - q2)**2*np.cos(q2)
            - 1848000*y*np.cos(q1)
            - 492800*ydot*np.cos(q1)
            - 170100*np.sin(q1 - q2)*np.sin(q1)**3
            - 66420*np.sin(q1 - q2)**3*np.sin(q1)
            + 8659200*y*np.cos(q1 - q2)
            + 2309120*ydot*np.cos(q1 - q2)
            # (... truncated for readability ...)
        )
    )

    denom = 22960000 * (np.cos(q1 - q2)*np.sin(q1) - np.sin(q1 - q2)*np.cos(q1))
    u1 = num1 / denom

    num2 = (
        -3 * (
            369600*np.cos(q1)
            - 4189500*np.cos(q1 - q2)**2*np.sin(q1)**2
            + 1077300*np.sin(q1 - q2)**2*np.cos(q1)**2
            - 56700*np.sin(q1 - q2)**2*np.sin(q1)**2
            + 56700*np.cos(q1)**2
            + 18900*np.cos(q1 - q2)*np.cos(q1)
            - 649440*np.cos(q1 - q2)*np.cos(q2)
            - 220500*np.sin(q1)**2
            # (... truncated for brevity ...)
        )
    )

    u2 = num2 / denom

    return u1, u2

def controller(model, data):
    # """Lie derivative-based controller for three-link walker"""
    
    # # Get current state
    q = data.qpos.copy()
    dq = data.qvel.copy()

    q1 = q[2]
    q2 = -q[3]
    q3 = q[3] + q[4]
    q1dot = dq[2]
    q2dot = -dq[3]
    q3dot = dq[3] + dq[4]
    y = q[1]
    ydot = dq[1]

    u1, u2 = compute_u(q1, q2, q1dot, q2dot, y, ydot)

    data.ctrl[3] = -u1  # pin2_motor
    data.ctrl[4] = -u2  # pin3_motor
        
    # # Get link lengths from model
    # l1 = link_lengths['geom_1']['length']
    # l2 = link_lengths['geom_2']['length']
    # l3 = link_lengths['geom_3']['length']
    
    # # Define model parameters
    # params = {
    #     'l1': l1, 'l2': l2, 'l3': l3,
    #     'm1': 1.0, 'm2': 1.0, 'm3': 1.0,
    #     'I1': 0.1, 'I2': 0.1, 'I3': 0.1,
    #     'd1': 0.5, 'd2': 0.375, 'd3': 0.375,
    #     'pCOMy_d': 2.0
    # }
    
    # # Create state object
    # state = states.States(q, dq, params)
    
    # # Debug print state information
    # state.debug_print()
    
    # # Use symbolic functions with state object
    # h = sym.h_func(state)
    # Jst = sym.Jst_func(state)
    # Jstdot = sym.Jstdot_func(state)
    # Jh = sym.Jh_func(state)
    # d2h = sym.d2h_func(state)
    
    # print(f"h = {h}")
    # print(f"Jst = {Jst}")
    
    # # Get system matrices from MuJoCo
    # D = np.zeros((5, 5))
    # mujoco.mj_fullM(model, D, data.qM)
    
    # # Get forces from MuJoCo
    # # For simplicity, use qfrc_bias which contains all bias forces
    # # In the standard form: D*ddq + C*dq + G = B*u
    # # where C*dq + G = qfrc_bias
    # bias_forces = data.qfrc_bias.copy()
    
    # # Input matrix B (assuming 2 inputs for the 2 virtual constraints)
    # B = np.zeros((5, 2))
    # B[2, 0] = 1  # First input affects q4
    # B[3, 1] = 1  # Second input affects q5

    # # B matrix:
    # # 0	0
    # # 0	0
    # # 1	0
    # # 0	1
    # # 0	0
    
    # # Jstdot already computed above
    
    # # Drift and input vector fields
    # f = np.concatenate([dq, np.linalg.solve(D, -bias_forces)])
    # g1 = np.concatenate([np.zeros((5, 2)), np.linalg.solve(D, B)])
    # g2 = np.concatenate([np.zeros((5, 2)), np.linalg.solve(D, Jst.T)])
    
    # # Jh and d2h already computed above
    
    # # Lie derivatives
    # Lfh = Jh @ dq
    # Lf2h = np.zeros(2)
    # for i in range(2):
    #     Lf2h[i] = d2h[i] @ f
    
    # Lg_1Lfh = np.zeros((2, 2))
    # Lg_2Lfh = np.zeros((2, 2))
    # for i in range(2):
    #     Lg_1Lfh[i] = d2h[i] @ g1
    #     Lg_2Lfh[i] = d2h[i] @ g2

    # print(f"Lfh = {Lfh}")
    # print(f"Lf2h = {Lf2h}")
    # print(f"Lg_1Lfh = {Lg_1Lfh}")
    # print(f"Lg_2Lfh = {Lg_2Lfh}")
    # print(f"B = {B}")
    # print(f"Jst = {Jst}")
    # print(f"Jstdot = {Jstdot}")
    # print(f"bias_forces = {bias_forces}")
    
    # # Controller gains
    # Kp = 50
    # Kd = 10
    
    # # Control input
    # v = -Kp * h - Kd * Lfh
    
    # # Constrained dynamics
    # LHS = np.block([[D, -B, -Jst.T],
    #                 [np.zeros((2, 5)), Lg_1Lfh, Lg_2Lfh],
    #                 [Jst, np.zeros((2, 2)), np.zeros((2, 2))]])
    
    # RHS = np.concatenate([-bias_forces, v - Lf2h, -Jstdot @ dq])
    
    # sol = np.linalg.solve(LHS, RHS)
    
    # ddq = sol[:5]  # accelerations
    # u = sol[5:7]   # control inputs
    
    # # Apply control to the appropriate actuators
    # if model.nu >= 2:
    #     # Map control inputs to actuators
    #     # u[0] -> pin2_motor, u[1] -> pin3_motor
    #     data.ctrl[3] = u[0]  # pin2_motor
    #     data.ctrl[4] = u[1]  # pin3_motor
        
    #     # Set other actuators to zero
    #     data.ctrl[0] = 0  # torso_x_motor
    #     data.ctrl[1] = 0  # torso_z_motor  
    #     data.ctrl[2] = 0  # torso_rot_motor
        
    #     # Debug output
    #     print(f"h = {h}, Lfh = {Lfh}, u = {u}")


with mujoco.viewer.launch_passive(model, data) as viewer:
    # Option 1: Set camera properties manually
    viewer.cam.distance = 7.0      # Distance from target
    viewer.cam.azimuth = 45.0      # Horizontal angle (0-360 degrees)
    viewer.cam.elevation = -30.0   # Vertical angle (-90 to 90 degrees)
    viewer.cam.lookat[:] = [0, 0, 3]  # Point to look at [x, y, z]
    
    # Option 2: Use predefined cameras from XML
    # viewer.cam.fixedcamid = model.camera("side_view").id  # Use side_view camera
    # viewer.cam.fixedcamid = model.camera("top_view").id   # Use top_view camera
    
    # while viewer.is_running():
    #     viewer.sync()
    #     time.sleep(model.opt.timestep)


    print("Starting simulation...")
    step_count = 0
    while viewer.is_running():
        # Apply controller
        controller(model, data)
        
        # Step simulation
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

