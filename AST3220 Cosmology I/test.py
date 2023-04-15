import numpy as np
import scipy.constants as sc

# defining constants
h = 0.7                  # Hubble constant
H0 = 100*h               # Hubble parameter [km s^-1 Mpc^-1]
Mpc = 3.086e22 # [m]
H0_scaled = H0*1e3/Mpc # [m s^-2 m^-1]
G = sc.G                 # Newtonian constant of gravitation [m^3 kg^-1 s^-2]
T0 = 2.72548             # temperature today of CMB [K]
N_eff = 3
c = sc.c                 # speed of light in vacuum [m s^-1]
k = sc.k                 # Boltzmann constant [J K^-1]
hbar = sc.hbar           # Planck constant [J Hz^-1]
Omega_r0 = ((8 * np.pi**3 * G * (k*T0)**4 ) / (45 * H0_scaled**2 * hbar**3 * c**5)) * (1 + N_eff * (7/8) * (4/11)**(4/3))

def age_of_universe(T):
    """
    T = temperature [K]
    """
    t = T0**2 / (T**2 * 2 * H0_scaled * np.sqrt(Omega_r0))

    return t

print(age_of_universe(1e10))
print(age_of_universe(1e9))
print(age_of_universe(1e8))
