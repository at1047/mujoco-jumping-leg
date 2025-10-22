import numpy as np

class States:
    def __init__(self, q = np.zeros([5,1]), dq = np.zeros([5,1]), params = None):
        self.x = q[0]
        self.y = q[1]
        self.q1 = q[2]
        self.q2 = -q[3]
        self.q3 = q[3] + q[4]
        self.dx = dq[0]
        self.dy = dq[1]
        self.dq1 = dq[2]
        self.dq2 = -dq[3]
        self.dq3 = dq[3] + dq[4]

        self.l1 = params['l1'] if params else None
        self.l2 = params['l2'] if params else None
        self.l3 = params['l3'] if params else None

        self.m1 = params['m1'] if params else None
        self.m2 = params['m2'] if params else None
        self.m3 = params['m3'] if params else None

        self.I1 = params['I1'] if params else None
        self.I2 = params['I2'] if params else None
        self.I3 = params['I3'] if params else None

        self.d1 = params['d1'] if params else None
        self.d2 = params['d2'] if params else None
        self.d3 = params['d3'] if params else None

        self.pCOMy_d = params['pCOMy_d'] if params else None
    
    def unpack(self):
        """
        Unpack all state data and return as separate outputs
        """
        return (self.x, self.y, self.q1, self.q2, self.q3,
                self.dx, self.dy, self.dq1, self.dq2, self.dq3,
                self.l1, self.l2, self.l3,
                self.m1, self.m2, self.m3,
                self.I1, self.I2, self.I3,
                self.d1, self.d2, self.d3,
                self.pCOMy_d)
    
    def debug_print(self):
        """
        Print debug information about the state
        """
        print("=== STATES DEBUG ===")
        print(f"Position states: x={self.x:.3f}, y={self.y:.3f}, q1={self.q1:.3f}, q2={self.q2:.3f}, q3={self.q3:.3f}")
        print(f"Velocity states: dx={self.dx:.3f}, dy={self.dy:.3f}, dq1={self.dq1:.3f}, dq2={self.dq2:.3f}, dq3={self.dq3:.3f}")
        print(f"Link lengths: l1={self.l1:.3f}, l2={self.l2:.3f}, l3={self.l3:.3f}")
        print(f"Masses: m1={self.m1:.3f}, m2={self.m2:.3f}, m3={self.m3:.3f}")
        print(f"Inertias: I1={self.I1:.3f}, I2={self.I2:.3f}, I3={self.I3:.3f}")
        print(f"CoM distances: d1={self.d1:.3f}, d2={self.d2:.3f}, d3={self.d3:.3f}")
        print(f"Desired CoM y: pCOMy_d={self.pCOMy_d:.3f}")
        print("==================")