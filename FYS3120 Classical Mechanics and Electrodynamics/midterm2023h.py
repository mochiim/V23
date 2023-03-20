import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use("seaborn")       # aestethic for plotting
plt.rcParams["font.size"] = 16 # font size of plots
plt.rcParams["lines.linewidth"] = 2  # line width of plots

m1 = 7000   # mass of counterweight [kg]
m2 = 15     # mass of mini-cow [kg]
l1 = 1.5    # [m]
l2 = 10     # [m]
g = 9.81    # gravitational acceleration [m/s^2]

# define some new variables for convenience
I = m1*l1**2 + m2*l2**2
MR = m1*l1 - m2*l2

t = np.linspace(-1e5, 1e5, 1000)
theta_i = 5*np.pi/6     # initial condition for angle theta
p_i = 0                 # initial condition for momentum

theta_f = np.linspace(0, 2*np.pi, 1000)
L_max = 2*(m1/m2)*l1*(1 - np.cos(theta_i))

def hamiltons_eq(theta):
    thetadot = p/I
    pdot = - MR*g*np.sin(theta)
    return [thetadot, pdot]

sol = solve_ivp(hamiltons_eq, t, [theta_i, p_i])
theta = sol[0, :]
p = sol[1, :]
