import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid

plt.style.use("seaborn")
plt.rcParams["font.size"] = 16 # font size of plots

def power_law_potential(N, parameters):
    x1, x2, x3, lmbda = parameters

    dx1 = - 3*x1 + (np.sqrt(6)/2)*lmbda*x2**2 + .5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = - (np.sqrt(6)/2)*lmbda*x1*x2 + .5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = - 2*x3 + .5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dlmbda = - np.sqrt(6)*lmbda**2*x1

    differentials = np.array([dx1, dx2, dx3, dlmbda])

    return differentials


def exponential_potential(N, parameters):
    x1, x2, x3 = parameters

    lmbda = 3/2
    dx1 = - 3*x1 + (np.sqrt(6)/2)*lmbda*x2**2 + .5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = - (np.sqrt(6)/2)*lmbda*x1*x2 + .5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = - 2*x3 + .5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)

    differentials = np.array([dx1, dx2, dx3])

    return differentials

def integration(function, init):

    solve = solve_ivp(function, z_span, init, method = 'RK45', rtol = 1e-8, atol = 1e-8, dense_output = True)

    exes = solve.sol(N)

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

variables_power = integration(power_law_potential, np.array([5e-5, 1e-8, .9999, 1e9]))
variables_exp = integration(exponential_potential, np.array([0, 5e-13, .9999]))

labels_power = [r"$\Omega_m$", r"$\Omega_\phi$", r"$\Omega_r$", r"$w_\phi$"]

"""
for i in range(4):
    plt.plot(z, variables_power[i])

plt.xscale("log")
plt.legend(labels_power)
plt.show()
"""

def Hubble_parameter(omega_m0, omega_phi0, omega_r0, EoS):
    w_phi = EoS
    integ = cumulative_trapezoid(3 * (1 + w_phi), N, initial = 0)
    H = np.sqrt(omega_m0 * np.exp(-3 * N) + omega_r0 * np.exp(-4 * N) + omega_phi0 * np.exp(integ))
    return H

H_exp = Hubble_parameter(variables_exp[0], variables_exp[1], variables_exp[2], variables_exp[3])
H_power = Hubble_parameter(variables_power[0], variables_power[1], variables_power[2], variables_power[3])
omega_m0CDM = .3
H_CDM = np.sqrt(omega_m0CDM * np.exp(-3 * N) + (1 - omega_m0CDM))
plt.plot(z, H_exp)
plt.plot(z, H_power)
plt.plot(z, H_CDM)
plt.xscale("log")
plt.yscale("log")
plt.show()
