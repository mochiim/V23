
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson, quad
from scipy.interpolate import CubicSpline
from tabulate import tabulate



""" Density parameters and equation of state """

def power_law_potential(N, parameters):
    """
    KOMMENTER
    """
    x1, x2, x3, lmbda = parameters

    dx1 = - 3*x1 + (np.sqrt(6)/2)*lmbda*x2**2 + .5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = - (np.sqrt(6)/2)*lmbda*x1*x2 + .5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = - 2*x3 + .5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dlmbda = - np.sqrt(6)*lmbda**2*x1

    return dx1, dx2, dx3, dlmbda


def exponential_potential(N, parameters):
    x1, x2, x3 = parameters

    lmbda = 3/2
    dx1 = - 3*x1 + (np.sqrt(6)/2)*lmbda*x2**2 + .5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = - (np.sqrt(6)/2)*lmbda*x1*x2 + .5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = - 2*x3 + .5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)

    return dx1, dx2, dx3

def integration(function, init):
    """
    A function which solves a differential equation using solve_ivp for a
    function [argument: function] and its initial conditions at z = 2e7
    [argument: init]. This results in x1, x2 and x3 which can be used to find
    density parameters and EoS parameter.
    """
    # ODE solver
    solve = solve_ivp(function, N_span, init, method = 'RK45', rtol = 1e-8, atol = 1e-8, dense_output = True)

    # solve for x1, x2 and x3
    exes = solve.sol(N)

    # unpack exes and asign names
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

# we define a new variable N for easier calculation
N_span = [-np.log(1 + 2e7), 0]
steps = 1000
N = np.linspace(-np.log(1 + 2e7), 0, steps)

# convert N back to z
z = np.exp(- N) - 1

# computation of density parameter and EoS for two Quintessence models
variables_power = integration(power_law_potential, np.array([5e-5, 1e-8, .9999, 1e9]))
variables_exp = integration(exponential_potential, np.array([0, 5e-13, .9999]))

""" Hubble parameter """

def Hubble_parameter(omega_m0, omega_phi0, omega_r0, EoS):
    w_phi = EoS
    integration = cumulative_trapezoid(3*(1 + np.flip(w_phi)), N, initial = 0) # initial 0 to avoid loss of one element
    H = np.sqrt(omega_m0 * np.exp(-3 * N) + omega_r0 * np.exp(-4 * N) + omega_phi0 * np.exp(np.flip(integration)))
    return H

H_exp = Hubble_parameter(variables_exp[0][-1], variables_exp[1][-1], variables_exp[2][-1], variables_exp[3])
H_power = Hubble_parameter(variables_power[0][-1], variables_power[1][-1], variables_power[2][-1], variables_power[3])
omega_m0CDM = .3
H_CDM = np.sqrt(omega_m0CDM * np.exp(-3 * N) + (1 - omega_m0CDM))

""" Age of the universe """
def age_of_universe(Hubble_parameter):
    t_0 = simpson(1/Hubble_parameter, N)
    return t_0


""" Luminosity distance """

def luminosity_distance(Hubble_parameter):
    """
    Computing the luminosity distance for a given Hubble parameter
    """
    search = np.logical_and(N <= - np.log(3) + 1e-2, N  >= - np.log(3) - 1e-2)
    idx = np.where(search == True)[0][0]
    N_reduced = N[idx:]
    H = Hubble_parameter[idx:]
    integration = cumulative_trapezoid(np.exp(- N_reduced)/H, N_reduced, initial = 0)
    dL = np.exp(- N_reduced)*np.flip(integration)

    z_dL = z[idx:]

    return z_dL, dL

""" problem 12 """

z_dL, dL_power = luminosity_distance(H_power)  # unitless
z_dL, dL_exp = luminosity_distance(H_exp)      # unitless

""" problem 13 """
z_data, dL_data, error_data = np.loadtxt('/Users/rebeccanguyen/Documents/GitHub/V23/AST3220 Cosmology I/sndata.txt', skiprows=5, unpack=True)

h = 0.7

# luminosity distance converted to units of length
dL_power_adjusted = dL_power*(3/h)   # [Gpc]
dL_exp_adjusted = dL_exp*(3/h)       # [Gpc]


def chisquared(model):
    z_dL, dL = model
    chi = 0
    for i, z in enumerate(z_data):
        interpol = CubicSpline(np.flip(z_dL), np.flip(dL))
        value = np.interp(z, np.flip(z_dL), interpol(np.flip(z_dL)))
        chi += (value - dL_data[i])**2 / error_data[i]**2
    return chi

#chisquared()
print(chisquared([z_dL, dL_power_adjusted]))
print(chisquared([z_dL, dL_exp_adjusted]))

"""
interpol = CubicSpline(np.flip(z_dL), np.flip(dL_exp_adjusted))
value = np.interp(0.799, np.flip(z_dL), interpol(np.flip(z_dL)))
plt.scatter(0.799, value)
plt.plot(z_dL, np.flip(interpol)(z_dL))
plt.plot(z_data, dL_data)
plt.show()
"""
