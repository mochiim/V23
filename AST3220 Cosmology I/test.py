import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use("seaborn")
plt.rcParams["font.size"] = 16 # font size of plots

def power_law_potential(N, parameters):
    x1, x2, x3, lmbda = parameters

    dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dlmbda = -np.sqrt(6) * lmbda**2 * x1

    differentials = np.array([dx1, dx2, dx3, dlmbda])

    return differentials


def exponential_potential(N, parameters):
    x1, x2, x3 = parameters

    lmbda = 3/2
    dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)

    differentials = np.array([dx1, dx2, dx3])

    return differentials

def integration(function, init):

    solve = solve_ivp(function, z_span, init, method = 'RK45', rtol = 1e-8, atol = 1e-8, dense_output = True)

    exes = solve.sol(N)
    print(len(exes))

    x1 = exes[0, :]
    x2 = exes[1, :]
    x3 = exes[2, :]

    # densitiy parameters
    omega_m = 1 - x1**2 - x2**2 - x3**2
    omega_phi = x1**2 + x2**2
    omega_r = x3**2

    # EoS parameter
    w_phi = (x1**2 - x2**2) / (x1**2 + x2**2)

    return omega_m, omega_phi, omega_r, w_phi


z_span = [-np.log(1 + 2e7), 0]

steps = 1000
N = np.linspace(-np.log(1 + 2e7), 0, steps)


z = np.exp(- N) - 1

omega_m, omega_phi, omega_r, w_phi = integration(power_law_potential, np.array([5e-5, 1e-8, .9999, 1e9]))

plt.plot(z, omega_m)
plt.plot(z, omega_phi)
plt.plot(z, omega_r)
plt.plot(z, w_phi)
plt.xscale("log")
plt.tight_layout()
plt.show()
