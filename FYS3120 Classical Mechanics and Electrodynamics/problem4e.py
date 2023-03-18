
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.style.use("seaborn")       # aestethic for plotting
plt.rcParams["font.size"] = 16 # font size of plots
plt.rcParams["lines.linewidth"] = 2  # line width of plots

# defining constants
m = 1. # mass of pendulum in kg
b = 0.5 # drag constant in kg/s

# setting up meshgrid for x, p plotting window
x_list = np.linspace(-5, 5, 1000)
p_list = np.linspace(-5, 5, 1000)
x, p = np.meshgrid(x_list, p_list)

# defining omega according to task descripton
omega_0 = np.array([0.1, b/m/2, 2])


fig, ax = plt.subplots(3, figsize = (6, 8))

titles = [r"$\frac{b}{m} > 2m$",
          r"$\frac{b}{m} = 2m$",
          r"$\frac{b}{m} < 2m$"]
for i, omega in enumerate(omega_0):
    xdot = p/m
    pdot = -m*omega**2*x - (b/m)*p

    ax[i].streamplot(x, p, xdot, pdot, color = "black")
    ax[i].set_xlabel("x [m]")
    ax[i].set_ylabel(r"p $[kgm/s]$ ")
    ax[i].set_title(titles[i])

plt.tight_layout()
#plt.show()
