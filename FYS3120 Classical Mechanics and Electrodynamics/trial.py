import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 1. # mass of pendulum in kg
b = 0.5 # drag constant in kg/s

xlim = 5 #
plim = 5 #

# Make plotting grid
xlist  = np.linspace(-xlim, xlim, 500)
plist = np.linspace(-plim, plim, 500)
x, p = np.meshgrid(xlist, plist)

omega_0 = (0.1, b/m/2., 2.)
print("b/m:", b/m)
print("2\omega:", 2*omega_0[0], 2*omega_0[1], 2*omega_0[2])

titles = (r"$\frac{b}{m} > 2\omega$",\
          r"$\frac{b}{m} = 2\omega$",\
          r"$\frac{b}{m} < 2\omega$")

plt.figure()
for (i, omega) in enumerate(omega_0):
    # Finding the phase space plots.
    xdot = p/(m)
    pdot = -(b/m)*p - m*omega**2*x

    # Plot phase space using streamplot-function.
    # streamplot uses (x,y) coordinates and their deriatives as input

    pos = int(221 + i)
    plt.subplot(pos)
    plt.title(titles[i])
    plt.ylabel(r'$p$ [kg m s$^{-1}$]')
    plt.streamplot(x, p, xdot, pdot, density = 0.9, color = "black")
    plt.xlabel(r'$x$ [m]')

plt.tight_layout()
#plt.show()
#plt.savefig('phase_space.png')
