import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson

'''
Settings for plots
'''
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 12

def differentials_powerlaw(N, variables):
	'''
	This function returns the differentials of
	x_i, to be used by solve_ivp. Variables
	are x1, x2, x3, and lambda.
	'''
	x1, x2, x3, lmbda = variables

	dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dlmbda = -np.sqrt(6) * lmbda**2 * x1

	diffs = np.array([dx1, dx2, dx3, dlmbda])

	return diffs


def differentials_exponential(N, variables):
	'''
	Same as for the power law, except that lambda is constant.
	'''
	lmbda = 3 / 2
	x1, x2, x3 = variables

	dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)

	diffs = np.array([dx1, dx2, dx3])

	return diffs


def integrate(potential_type):
	'''
	Function to integrate the system. The argument potential_type has
	to be either 'power-law' or 'exponential'.
	'''

	if potential_type == 'power-law':

		# Setting initial values at z = 2e7, ie. N = -ln(1 + 2e7)
		x1_0 = 5e-5
		x2_0 = 1e-8
		x3_0 = 0.9999
		lmbda_0 = 1e9

		initial_variables = np.array([x1_0, x2_0, x3_0, lmbda_0])

		sol = solve_ivp(differentials_powerlaw, z_range, initial_variables, method='RK45', rtol=1e-9, atol=1e-9, dense_output=True)

	if potential_type == 'exponential':

		# Initial values for exponential potential
		x1_0 = 0.
		x2_0 = 5e-13
		x3_0 = 0.9999

		initial_variables = np.array([x1_0, x2_0, x3_0])

		sol = solve_ivp(differentials_exponential, z_range, initial_variables, method='RK45', rtol=1e-9, atol=1e-9, dense_output=True)

	variables = sol.sol(z)

	# Extracting the solutions from solve_ivp
	x1 = variables[0, :]
	x2 = variables[1, :]
	x3 = variables[2, :]

	# Saving x1, x2, and x3 to be used later
	np.save('x1', x1)
	np.save('x2', x2)
	np.save('x3', x3)

	# Calculating the parameters
	Omega_m = 1 - x1**2 - x2**2 - x3**2
	Omega_phi = x1**2 + x2**2
	Omega_r = x3**2

	# Calculating the EoS parameter
	w_phi = (x1**2 - x2**2) / (x1**2 + x2**2)

	return Omega_m, Omega_phi, Omega_r, w_phi

z_range = [-np.log(1 + 2e7), 0]

steps = 1000
z = np.linspace(0, 2e7, steps)


Omega_m = np.zeros((2, steps))
Omega_phi = np.zeros((2, steps))
Omega_r = np.zeros((2, steps))
w_phi = np.zeros((2, steps))

Omega_m[0, :], Omega_phi[0, :], Omega_r[0, :], w_phi[0, :] = integrate('power-law')
Omega_m[1, :], Omega_phi[1, :], Omega_r[1, :], w_phi[1, :] = integrate('exponential')

plt.plot(z, Omega_m[0, :])
plt.plot(z, Omega_phi[0, :])
plt.plot(z, Omega_r[0, :])
plt.plot(z, w_phi[0, :])
plt.show()
