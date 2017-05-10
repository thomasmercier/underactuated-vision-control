import numpy as np

class DoublePendulum:
    
    # inertial parameters
    m1 = 0
    m2 = 0
    IG = 0
    IH = 0
    # geometric parameters
    a = 0
    b = 0
    c = 0
    d = 0
    # dynamic state (2 angles, 2 angular velocities)
    state = np.empty(4);

    def __init__(self, inert, geom):
        m1 = inert[0]
        m2 = inert[1]
        IG = inert[2]
        IH = inert[3]
        a = geom[0]
        b = geom[1]
        c = geom[2]
        d = geom[3]

    def f(self):
        adot = self.state[2]
        bdot = self.state[3]
        cosa = np.cos(self.state[0])
        sina = np.sin(self.state[0])
        cosb = np.cos(self.state[1])
        sinb = np.sin(self.state[1])
        cosab = np.cos(self.state[0]-self.state[1])
        sinab = np.sin(self.state[0]-self.state[1])
        K = m1*c**2 + m2*a**2 + IG
        L = m2*b**2 + IH
        M = m2*a*b
        g = 9.81
        P = -g*(m1*c+m2*a)
        Q = -g*m2
        den = L*K - L**2*cosab
        addot = P*L*sina-M*Q*sinb*cosab-M*L*bdot**2*sinab-M**2*adot**2*sinab*cosab
        addot = addot / den
        bddot = K*Q*sinb-M*P*sina*cosab+M*K*adot**2*sinab+M**2*bdot**2*sinab*cosab
        bddot = bddot / den
        return np.array([adot, bdot, addot, bddot])


