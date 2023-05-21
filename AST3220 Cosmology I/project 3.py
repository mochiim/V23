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
psi = lampbda psi: phi / E_P
v = lambda psi: (3 * hbar * c ** 5) / (8 * np.pi * G) * (psi / phi_init) ** 2
    


