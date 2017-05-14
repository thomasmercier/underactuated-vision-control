import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class FiniteStatePendulum:

    def __init__(self, nState):
        self.omega = 1
        self.nState = nState
        self.angleBins = np.linspace(0, np.pi, self.nState, \
            endpoint=False)[1:self.nState]
        self.velocityBins = np.linspace(-2*self.omega, \
            2*self.omega, self.nState-1)
        self.dt = 0.1
        self.continuousState = np.empty(2)
        self.discreteState = 0
        self.feeback = 0

    def nextState(self, feedback):
        self.feedback = feedback
        def f(state, t):
            return [state[1], -self.omega**2*np.sin(state[0]) + feedback]
        self.continuousState = integrate.odeint(f, self.state, [0, self.dt])[1,:]
        return self.makeDiscreteState()

    def scaling1(x):
        return np.cos(x/np.pi)

    def scaling2(x):
        return 2*np.sqrt(x-np.pow(x,2))

    def makeDiscreteState(self):
        self.continuousState[0] = self.continuousState[0] % (2*np.pi)
        iAngle = np.digitize(self.continuousState[0], self.angleBins)
        iVelocity = np.digitize(self.continuousState[1], self.velocityBins)
        self.discreteState = iAngle + self.nState*iVelocity
        return self.discreteState

    def randomInit(self):
        angle = 2*np.pi*np.random.rand()
        velocity = 8*self.omega*(np.random.rand()-0.5)
        self.continuousState = [angle, velocity]
        return self.makeDiscreteState()

class ControlledPendulum:

    def __init__(self):
        self.nAction = 21
        self.nState = 21
        self.pendulum = FiniteStatePendulum(self.nState)
        self.Q = np.zeros([self.nState**2, self.nAction])
        self.buildReward()
        self.buildAction()
        self.policy = np.zeros([self.nState**2], dtype=np.int8)

    def buildReward(self):
        self.reward = np.empty([self.nState**2, self.nState**2])
        self.pendulum.state = [0, 0]
        k0 = self.pendulum.makeDiscreteState()
        for i in range(self.nAction**2):
            for j in range(self.nAction**2):
                if j==k0:
                    self.reward[i,j] = 10
                else:
                    self.reward[i,j] = 0

    def buildAction(self):
        torqueMax = 10
        self.action = np.linspace(-torqueMax, torqueMax, self.nAction)

    def alpha(self, n):
        # TO DO
        return 0.5

    def epsilon(self, n):
        # TO DO
        return 0.5

    def learn(self):
        nInit = 100
        nRun = 1000
        gamma = 0.8
        for k in range(nInit):
            print('k='+str(k))
            state = self.pendulum.randomInit();
            for i in range(nRun):
                if np.random.rand() > self.epsilon(i):
                    iAction = np.argmax(self.Q[state,:])
                else:
                    iAction = np.random.randint(self.nAction)
                self.pendulum.nextState(self.action[iAction])
                nextState = self.pendulum.discreteState
                self.Q[state, iAction] += self.alpha(i) * \
                    ( self.reward[state, nextState] \
                      + gamma*np.amax(self.Q[nextState,:]) \
                      - self.Q[state, iAction] )
                state = nextState
        for i in range(self.nState**2):
            self.policy[i] = np.argmax(self.Q[state,:])

    def go(self):
        nRun = 1000
        t = self.pendulum.dt*np.arange(nRun)
        theta = np.empty(nRun)
        state = self.pendulum.randomInit();
        for i in range(nRun):
            theta[i] = self.pendulum.continuousState[0]
            print( str(theta[i])+' -- '+str(self.action[self.policy[state]]))
            state = self.pendulum.nextState( self.action[self.policy[state]] )
        print(self.Q)
        plt.plot(t, theta)
        plt.show()
