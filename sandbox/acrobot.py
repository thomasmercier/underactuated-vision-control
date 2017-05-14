import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class Acrobot:

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
        self.m1 = inert[0]
        self.m2 = inert[1]
        self.IG = inert[2]
        self.IH = inert[3]
        self.a = geom[0]
        self.b = geom[1]
        self.c = geom[2]
        self.d = geom[3]

    def setState(self, state):
        self.state = state;

    def Gamma(self):
	return -0.5*self.state[3]

    def f(self):
        adot = self.state[2]
        bdot = self.state[3]
        cosa = np.cos(self.state[0])
        sina = np.sin(self.state[0])
        cosb = np.cos(self.state[1])
        sinb = np.sin(self.state[1])
        cosab = np.cos(self.state[0]-self.state[1])
        sinab = np.sin(self.state[0]-self.state[1])
        K = self.m1*self.c**2 + self.m2*self.a**2 + self.IG
        L = self.m2*self.b**2 + self.IH
        M = self.m2*self.a*self.b
        g = 9.81
        P = -g*(self.m1*self.c+self.m2*self.a)
        Q = -g*self.m2
        den = L*K - L**2*cosab
	addot = P*L*sina-M*Q*sinb*cosab-M*L*bdot**2*sinab-M**2*adot**2*sinab*cosab
        addot = addot / den
        bddot = K*Q*sinb-M*P*sina*cosab+M*K*adot**2*sinab+M**2*bdot**2*sinab*cosab
	bddot = bddot + K*self.Gamma()
        bddot = bddot / den
        return np.array([adot, bdot, addot, bddot])

    def f2(self, state, t):
        self.setState(state)
        return self.f()

    def show(self, state, T, dt):
        # mostly copied from
        # https://matplotlib.org/examples/animation/double_pendulum_animated.html
        t = np.arange(0.0, T, dt)
        y = integrate.odeint(self.f2, state, t)
        print(y)

        x1 = self.a*np.sin(y[:, 0])
        y1 = -self.a*np.cos(y[:, 0])
        x2 = self.b*np.sin(y[:, 1]) + x1
        y2 = -self.b*np.cos(y[:, 1]) + y1

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]
            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i*dt))
            return line, time_text

        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                                      interval=25, blit=True, init_func=init)

        # ani.save('double_pendulum.mp4', fps=15)
        plt.show()
