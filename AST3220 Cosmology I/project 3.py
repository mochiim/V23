import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ast

""" Task d """
hbar = ast.hbar.value                 # reduced Planck constant [J/s]
G = ast.G.value                       # gravitational constant [m^3/ (kg s^2)]
c = ast.c.value                       # speed of light in vacuum [m/s]

E_P = np.sqrt((hbar * c ** 5) / G)    # Planck energy

e_folds = 500
phi_init = E_P * np.sqrt((e_folds + 1/2 ) / (2*np.pi))    # initial value for scalar field 
print(f"Initial value for the field that will give 500 e-folds of inflation is ", "{:e}".format(phi_init))

""" Task e"""
v = lambda 
psi = lambda psi: phi / E_P
dv_dpsi = lambda psi: 3 * hbar * c^2 * psi / (4 * np.pi * G * phi_i **2)
dpsi_dtau = lambda h, dpsi, dv: - 3 * h * dpsi - dv
    

def solve(N = 10):
    dtau = 1e-3
    tau = np.linspace(0, N, 1000)

    # array for storage
    h = np.zeros(N)
    psi = np.zeros(N)
    dpsi = np.zeros(N)
    ln_a_ai = np.zeros(N)
    
    # initial values

    for i in range(N):
        ln_a_ai[i + 1] = ln_a_ai + h[i] * dtau
        dpsi[i + 1] = dpsi[i] + psi_double_dot[i] * dtau
        psi[i + 1] = psi[i] + dpsi[i] * dtau
        h[i + 1] = h[i]
