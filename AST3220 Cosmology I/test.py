import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
import astropy.constants as const
from scipy import interpolate


c = const.c.cgs.value         # light speed
k_B = const.k_B.cgs.value     # Boltzmann constant
G = const.G.cgs.value         # Newton's gravitational constant
h_bar = const.hbar.cgs.value  # reduced Planck constant
m_e = const.m_e.cgs.value     # electron mass
m_p = const.m_p.cgs.value     # proton mass
m_n = const.m_n.cgs.value     # neutron mass

h = 0.7                # hubble constant
N_eff = 3              # neutrino species
T0 = 2.725             # [K] today's temperature of the CMB
H0 = 100 * h * 1e5 / 3.086e24  # Hubble parameter in CGS
tau = 1700             # [s] free neutron decay time


def dY_f(logT, Y):
        """
        Differential eqs for dYn/dlnT, dYp/dlnT from eqs (10), (11).
        Work out dY/dlnT for the first two weak reactions in table 2.
        """
        T = np.exp(logT) # [K] temperature
        a = 0 / T

        omega_r0 = (8 * np.pi**3 * G) / (90 * H0**2) \
                    * (k_B * T0)**4 / (h_bar**3 * c**5) \
                    * (2 + N_eff * 7/4 * (4/11)**(4/3))

        H = H0 * np.sqrt(omega_r0) * a**(-2)
        Y_n = Y[0]; Y_p = Y[1]


        dY_n = -1 / H * (Y_p * gamma_p_n(T) - Y_n * gamma_n_p(T))
        dY_p = -1 / H * (Y_n * gamma_n_p(T)- Y_p * gamma_p_n(T))

        return np.array([dY_n, dY_p])

def task_f():
        """Solve differential equations for dYn and dYp and plot the results.
        """
        # inial conditions for Y_n  and Y_p:
        T_i = 1e11
        Yn_Ti = (1 + np.exp( (m_n - m_p) \
                            * c**2 / k_B * T_i ) )**(-1)
        Yp_Ti = 1 - Yn_Ti

        solve = int.solve_ivp(dY_f, t_span=[np.log(100e9), np.log(0.1e9)],\
                y0=[Yn_Ti, Yp_Ti], method='Radau', rtol=1e-12, atol=1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]
        Y_p = solve.y[1]

        plt.plot(T, Y_p, label='p')
        plt.plot(T, Y_n, label='n')

        plt.show()






def gamma_n_p(T, q = 2.53):

        T_nu = (4/11)**(1/3) * T # From task b)
        Z_nu = m_e * c**2 / (k_B * T_nu)
        Z = m_e * c**2 / (k_B * T)

        f = lambda x: ((x + q)**2 * (x**2 - 1)**(1/2) * x) \
            / ((1 + np.exp(x * Z)) * (1 + np.exp( -(x + q) * Z_nu ))) \
            + ((x - q)**2 * (x**2 - 1)**(1/2) * x ) \
            / ((1 + np.exp(-x * Z)) * (1 + np.exp( (x - q) * Z_nu )))

        x = np.linspace(1, 100, 1000)
        dx = x[1] - x[0]
        return 1/tau * np.trapz(f(x), dx=dx)

def gamma_p_n(T, q=2.53):
        return gamma_n_p(T, -q)


task_f()
