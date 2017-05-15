import numpy as np
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt

def f1(x, t):
    w = 1
    Q = 20
    return [x[1], -w**2*np.sin(x[0])-w/Q*x[1]]

def f2(t, x):
    return f1(x, t)

T = 50
dt = 0.1
x0 = [0, 10]

t = np.arange(0,T,dt)
N = len(t)
x1 = odeint(f1, x0, t)

integr = ode(f2).set_integrator('vode', method='bdf')
integr.set_initial_value(x0)
x2 = np.empty([N,2])
x2[0,:] = x0
for i in range(1, N):
    x2[i,:] = integr.integrate(t[i])


plt.plot(t, x1[:,0])
plt.plot(t, x2[:,0])
plt.show()
