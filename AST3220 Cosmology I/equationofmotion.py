import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson, quad
from tabulate import tabulate

plt.style.use("seaborn")       # aestethic for plotting
plt.rcParams["font.size"] = 16 # font size of plots
plt.rcParams["lines.linewidth"] = 2  # line width of plots


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

""" problem 9 """
#labels_omega_w = [r"$\Omega_m$", r"$\Omega_\phi$", r"$\Omega_r$", r"$w_\phi$"]

#for i in range(4):
    #plt.plot(z, variables_power[i])
    #plt.plot(z, variables_exp[i])

#plt.title(r"$\Omega_m$, $\Omega_\phi$, $\Omega_r$ and $w_\phi$ for power law exponential")
#plt.title(r"$\Omega_m$, $\Omega_\phi$, $\Omega_r$ and $w_\phi$ for exponential exponential")
#plt.xscale("log"); plt.xlabel("z")
#plt.legend(labels_omega_w)
#plt.savefig("powerlaw.png")
#plt.savefig("exp.png")


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

""" problem 10 """
#labels_H = [r"$H(z)^{exp}$", r"$H(z)^{power}$", r"$H(z)^{\Lambda CDM}$"]
#plt.plot(z, H_exp, ls = "dotted")
#plt.plot(z, H_power, ls = (0, (5, 10)), color = "purple")
#plt.plot(z, H_CDM)
#plt.xscale("log"); plt.yscale("log")
#plt.xlabel("z"); plt.ylabel(r"$H(z)/H_0$")
#plt.title("Hubble parameter")
#plt.legend(labels_H)
#plt.savefig("Hubble_parameter.png")


""" Age of the universe """
def age_of_universe(Hubble_parameter):
    t_0 = simpson(1/Hubble_parameter, N)
    return t_0

""" problem 11 """
#t_exp = age_of_universe(H_exp)
#t_power = age_of_universe(H_power)
#t_CDM = age_of_universe(H_CDM)
#age = [["Quintessence: exponential", t_exp], ["Quintessence: power law", t_power], ["LCDM", t_CDM]]
#print(tabulate(age))


""" Luminosity distance """
z_dL = np.linspace(0, 2, 1000)

def luminosity_distance(Hubble_parameter):
    integration  = cumulative_trapezoid(1/Hubble_parameter, z_dL, initial = 0)
    dL = (1 + z_dL) * integration
    return dL

""" problem 12 """
#dL_power = luminosity_distance(H_power)
#dL_exp = luminosity_distance(H_exp)
#plt.plot(z_dL, dL_power)
#plt.plot(z_dL, dL_exp)


""" problem 13 """
z_data, dL_data, error_data = np.loadtxt('/Users/rebeccanguyen/Documents/GitHub/V23/AST3220 Cosmology I/sndata.txt', skiprows=5, unpack=True)

def chisquared():
    chi = 0

    for i in range(31):
        chi += (luminosity_distance(z_data[i]) - dL_data[i]**2) / error_data[i]**2
    return chi

chisquared()

#plt.plot(z_data, dL_data)
#for i in error_data:
#    plt.fill_between(z_data, dL_data + i, dL_data - i, alpha=0.2)
#plt.xlabel("z")
#plt.ylabel(r"$d_L$ [Gpc]")
#plt.title("Measured luminosity ditsance with assosiated errors")
#plt.savefig("sndata.png")

plt.show()
