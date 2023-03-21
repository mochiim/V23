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


theta_list = np.linspace(- 2*np.pi, 2*np.pi, 1000)
p_list = np.linspace(-1e5, 1e5, 1000)
theta, p = np.meshgrid(theta_list, p_list)


thetadot = p/I
pdot = - MR*g*np.sin(theta)
Hamiltonian = lambda theta, p: p**2/I - MR*g*np.cos(theta)
"""
fig, ax = plt.subplots(figsize = (8, 6))
strm = plt.streamplot(theta, p, thetadot, pdot, color = Hamiltonian(theta, p), cmap = "hot")
fig.colorbar(strm.lines, label = r"H($theta$, p)")
plt.xlabel("x [m]")
plt.ylabel(r"p $[kgm/s]$")
plt.title("Trebuchet")
#plt.savefig("g.png")
"""

plt.show()
