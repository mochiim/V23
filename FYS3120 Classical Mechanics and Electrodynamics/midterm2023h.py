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

t = np.linspace(0, 10, 10000)
theta_i = 5*np.pi/6             # initial condition for angle theta
p_i = 0                         # initial condition for momentum
init = np.array([theta_i, p_i]) # initial conditions

L_max = 2*(m1/m2)*l1*(1 - np.cos(theta_i))

def H(t, Y):
    theta, p = Y
    thetadot = p/I
    pdot = - MR*g*np.sin(theta)
    return [thetadot, pdot]

sol = solve_ivp(H, [0, 10], init, method = 'RK45', rtol = 1e-8, atol = 1e-8, dense_output = True)
Y = sol.sol(t)
theta, p = Y

thetadot = p/I

#v = -l2*thetadot*np.cos(theta)
v = thetadot*l2

L = (v**2 * np.sin(2*theta))/g

epsilon = L/L_max # range efficiency


idx = np.where(epsilon == np.max(epsilon))[0][0]
print(f"Maximum efficiency is {epsilon[idx]} at angle {theta[idx]}")

#plt.plot(theta, p)
plt.plot(theta, epsilon)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\epsilon_L$")
plt.text(epsilon[idx] + .5, theta[idx] + .5, "hei")
plt.title(r"Range efficiency for $\theta_i= 5\pi/6$")
#plt.savefig("h1.png")
plt.show()
