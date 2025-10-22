import sympy as sp
import numpy as np

# Calculating symbolic expressions


# q0, q1, q2, q3, q4= sp.symbols('q0 q1 q2 q3 q4', real=True)
# q_sym = sp.Matrix([q0, q1, q2, q3, q4])

# MATLAB-generated h function
def auto_h(state):
    """
    MATLAB-generated h function
    """
    # Unpack state data
    x, y, q1, q2, q3, dx, dy, dq1, dq2, dq3, l1, l2, l3, m1, m2, m3, I1, I2, I3, d1, d2, d3, pCOMy_d = state.unpack()
    
    t2 = np.cos(q1)
    t3 = np.sin(q1)
    t4 = m1 + m2 + m3
    t6 = -q2
    t7 = -x
    t5 = l1 * t3
    t8 = q1 + t6
    t9 = 1.0 / t4
    t10 = np.sin(t8)
    
    h = np.array([
        t5 + t7 + l2 * t10 + t9 * (-m2 * (t5 + t7 + d2 * t10) + m3 * (x - d3 * np.sin(q3)) + m1 * (x - d1 * t3)),
        -pCOMy_d + t9 * (m3 * (y + d3 * np.cos(q3)) + m1 * (y - d1 * t2) - m2 * (-y + l1 * t2 + d2 * np.cos(t8)))
    ])
    
    return h

# Create lambda function for h
h_func = auto_h

# MATLAB-generated Jst function
def auto_Jst(state):
    """
    MATLAB-generated Jst function
    """
    # Unpack state data
    x, y, q1, q2, q3, dx, dy, dq1, dq2, dq3, l1, l2, l3, m1, m2, m3, I1, I2, I3, d1, d2, d3, pCOMy_d = state.unpack()
    
    t2 = -q2
    t3 = q1 + t2
    t4 = np.cos(t3)
    t5 = np.sin(t3)
    t6 = l2 * t4
    t7 = l2 * t5
    
    Jst = np.array([
        [1.0, 0.0, 0.0, 1.0, -t6 - l1 * np.cos(q1)],
        [0.0, 1.0, t7 + l1 * np.sin(q1), t6, -t7]
    ]).reshape(2, 5)
    
    return Jst

# Create lambda function for Jst
Jst_func = auto_Jst

# MATLAB-generated Jstdot function
def auto_Jstdot(state):
    """
    MATLAB-generated Jstdot function
    """
    # Unpack state data
    x, y, q1, q2, q3, dx, dy, dq1, dq2, dq3, l1, l2, l3, m1, m2, m3, I1, I2, I3, d1, d2, d3, pCOMy_d = state.unpack()
    
    t2 = -q2
    t3 = -dq2
    t4 = q1 + t2
    t5 = dq1 + t3
    t6 = np.cos(t4)
    t7 = np.sin(t4)
    
    Jstdot = np.array([
        # [0.0, 0.0, 0.0, 0.0, dq1 * (l2 * t7 + l1 * np.sin(q1)) + l2 * t3 * t7],
        # [0.0, 0.0, 0.0, 0.0, dq1 * (l2 * t6 + l1 * np.cos(q1)) + l2 * t3 * t6],
        # [-l2 * t5 * t7, -l2 * t5 * t6, 0.0, 0.0, 0.0]
        [0.0,0.0,0.0,0.0,dq1*(l2*t7+l1*np.sin(q1))+l2*t3*t7,dq1*(l2*t6+l1*np.cos(q1))+l2*t3*t6,-l2*t5*t7,-l2*t5*t6,0.0,0.0]
    ]).reshape(2, 5)
    
    return Jstdot

# Create lambda function for Jstdot
Jstdot_func = auto_Jstdot

# MATLAB-generated Jh function
def auto_Jh(state):
    """
    MATLAB-generated Jh function
    """
    # Unpack state data
    x, y, q1, q2, q3, dx, dy, dq1, dq2, dq3, l1, l2, l3, m1, m2, m3, I1, I2, I3, d1, d2, d3, pCOMy_d = state.unpack()
    
    t2 = np.cos(q1)
    t3 = np.sin(q1)
    t5 = m1 + m2 + m3
    t6 = -q2
    t4 = l1 * t2
    t7 = q1 + t6
    t9 = 1.0 / t5
    t8 = np.cos(t7)
    t10 = np.sin(t7)
    t11 = l2 * t8
    
    Jh = np.array([
        [0.0,0.0,0.0,1.0,t4+t11-t9*(m2*(t4+d2*t8)+d1*m1*t2),t9*(m2*(d2*t10+l1*t3)+d1*m1*t3),-t11+d2*m2*t8*t9,-d2*m2*t9*t10,-d3*m3*t9*np.cos(q3),-d3*m3*t9*np.sin(q3)]
    ]).reshape(2, 5)
    
    return Jh

# MATLAB-generated d2h function
def auto_d2h__(state):
    """
    MATLAB-generated d2h function
    """
    # Unpack state data
    x, y, q1, q2, q3, dx, dy, dq1, dq2, dq3, l1, l2, l3, m1, m2, m3, I1, I2, I3, d1, d2, d3, pCOMy_d = state.unpack()
    
    t2 = np.cos(q1)
    t3 = np.cos(q3)
    t4 = np.sin(q1)
    t5 = np.sin(q3)
    t7 = m1 + m2 + m3
    t9 = -q2
    t6 = l1 * t2
    t8 = l1 * t4
    t10 = d1 * m1 * t2
    t11 = d1 * m1 * t4
    t12 = q1 + t9
    t14 = 1.0 / t7
    t13 = np.cos(t12)
    t15 = np.sin(t12)
    t16 = d2 * t13
    t17 = l2 * t13
    t18 = d2 * t15
    t19 = l2 * t15
    t20 = t6 + t16
    t21 = t8 + t18
    t24 = m2 * t14 * t18
    t25 = m2 * dq2 * t14 * t16
    t22 = m2 * t20
    t23 = m2 * t21
    t26 = -t24
    t27 = t11 + t23
    t28 = t10 + t22
    t29 = t19 + t26
    t30 = t14 * t27
    t31 = dq2 * t29
    
    d2h__ = np.array([
        [0.0,0.0,0.0,0.0,t31-dq1*(t8+t19-t30),-t25+dq1*t14*t28,-t31+dq1*t29,t25-m2*dq1*t14*t16,d3*m3*dq3*t5*t14,-d3*m3*dq3*t3*t14,0.0,0.0,0.0,1.0,t6+t17-t14*t28,t30,-t17+m2*t14*t16,t26,-d3*m3*t3*t14,-d3*m3*t5*t14]
    ]).reshape(2, 10)
    
    return d2h__

# Create lambda functions
Jh_func = auto_Jh
d2h_func = auto_d2h__