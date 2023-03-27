import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson, quad
from tabulate import tabulate
from scipy.interpolate import CubicSpline

plt.style.use("seaborn")       # aestethic for plotting
plt.rcParams["font.size"] = 16 # font size of plots
plt.rcParams["lines.linewidth"] = 2  # line width of plots


""" Density parameters and equation of state """

def power_law_potential(N, parameters):
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

    # unpack exes and assign names
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

# substitute N for z and flipping the integration interval for convenience
N_span = [-np.log(1 + 2e7), 0] # equiv. to [2e7, 0]
steps = 1000
N = np.linspace(-np.log(1 + 2e7), 0, steps)

# convert N back to z for plotting purposes
z = np.exp(- N) - 1

# computation of density parameter and EoS for two quintessence models: [omega_m, omega_phi, omega_r, w_phi]
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
    """
    Computing the dimensionless age of the Universe
    """
    t_0 = simpson(1/Hubble_parameter, N)
    return t_0


""" problem 11 """
t_exp = age_of_universe(H_exp)
t_power = age_of_universe(H_power)
t_CDM = age_of_universe(H_CDM)
age = [["Quintessence: exponential", t_exp], ["Quintessence: power law", t_power], ["LCDM", t_CDM]]
#print(tabulate(age))
"""
Terminal output:
-------------------------  --------
Quintessence: exponential  0.972921
Quintessence: power law    0.993174
LCDM                       0.964099
-------------------------  --------
"""


""" Luminosity distance """
# in this task, we are only interested in z = [0, 2]

def luminosity_distance(Hubble_parameter):
    """
    Computing the dimensionless luminosity distance for a given Hubble parameter (H/H0)
    """

    # finding index in N-array where N = -np.log(3), equivalent to z = 2
    search = np.logical_and(N <= - np.log(3) + 1e-2, N  >= - np.log(3) - 1e-2)
    idx = np.where(search == True)[0][0]

    # shortening N-array from [0, 2e7] to [0, 2], as well as Hubble parameter array
    N_reduced = N[idx:]
    H = Hubble_parameter[idx:]

    integration = cumulative_trapezoid(np.exp(- N_reduced)/H, N_reduced, initial = 0)
    dL = np.exp(- N_reduced)*np.flip(integration)

    z_dL = z[idx:]

    return z_dL, dL

""" problem 12 """

z_dL, dL_power = luminosity_distance(H_power)  # unitless
z_dL, dL_exp = luminosity_distance(H_exp)      # unitless

#plt.plot(z_dL, dL_power, label = "Power law potential")
#plt.plot(z_dL, dL_exp, label = "Exponential potential")
#plt.ylabel(r"$H_0d_L/c$")
#plt.xlabel("z")
#plt.title("Luminosity distance")
#plt.legend()
#plt.savefig("lum_dis.png")

""" Chi squared """

# reading data file
z_data, dL_data, error_data = np.loadtxt('/Users/rebeccanguyen/Documents/GitHub/V23/AST3220 Cosmology I/sndata.txt', skiprows=5, unpack=True)

h = 0.7 # given in task description

# luminosity distance for both quintessence models converted to units of length
dL_power_adjusted = dL_power*(3/h)   # [Gpc]
dL_exp_adjusted = dL_exp*(3/h)       # [Gpc]


def chisquared(model):
    z_dL, dL = model
    chi = 0
    for i, z in enumerate(z_data):
        # interpolating computed luminosity distance for both quintessence models
        interpol = CubicSpline(np.flip(z_dL), np.flip(dL))

        # finding d_L in interpolated array for a given z value from sndata.txt
        value = np.interp(z, np.flip(z_dL), interpol(np.flip(z_dL)))

        # computing chi squared
        chi += (value - dL_data[i])**2 / error_data[i]**2
    return chi

""" problem 13 """
chisquared_pwr = chisquared([z_dL, dL_power_adjusted])
chisquared_exp = chisquared([z_dL, dL_exp_adjusted])

chisqrt = [["Value of χ2 for power potential", chisquared_pwr], ["Value of χ2 for exponential potential", chisquared_exp]]
#print(tabulate(chisqrt))

"""
Terminal output:
-------------------------------------  -------
Value of χ2 for power potential        240.641
Value of χ2 for exponential potential  224.864
-------------------------------------  -------
"""

#plt.plot(z_data, dL_data, label = "Data")
#plt.plot(z_dL, luminosity_distance(H_CDM)[1]*(3/h), label = r"$\Lambda CDM$")

#for i in error_data:
#    plt.fill_between(z_data, dL_data + i, dL_data - i, alpha=0.2)

#plt.title("Measured luminosity distance with assosiated errors")

#plt.plot(z_dL, dL_exp_adjusted, label = "Exponential potential")
#plt.plot(z_dL, dL_power_adjusted, label = "Power law potential")
#plt.title(r"Data compared to quintessence and $\Lambda$CDM")

#plt.xlabel("z")
#plt.ylabel(r"$d_L$ [Gpc]")
#plt.legend()

""" Determine value of Ωm0 which provides the best fit """
def best_value():
    omegas = np.linspace(0, 1, 1000) # potential values

    # creating arrays for storing values
    chi = np.zeros(1000)
    omega_m0CDM = np.zeros(1000)

    for i in range(len(omegas)):
        H_CDM = np.sqrt(omegas[i] * np.exp(-3 * N) + (1 - omegas[i])) # Hubble parameter for a given omega
        z, dL = luminosity_distance(H_CDM)                            # luminosity distance [-]
        dL_new = dL*(3/h)                                             # luminosity distance [Gpc]
        chi[i] = chisquared([z, dL_new])
        omega_m0CDM[i] = omegas[i]

    return chi, omega_m0CDM

""" problem 14 """
chi, omegas = best_value()
#print(f"Lowest value from chi squared method {chi[np.argmin(chi)]: .3f}")
#print(f"Corresponding omega_m0LCDM value: {omegas[np.argmin(chi)]: .3f}")

"""
Terminal output:
Lowest value from chi squared method  32.960
Corresponding omega_m0LCDM value:  0.303
"""

#plt.show()
