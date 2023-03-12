import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

"""
This script will calculate the Gamow peak for all the reactions of the PP chain and CNO cycle
"""

plt.rcParams['font.size'] = 16 # font size of plots

# defining physical constants
e = const.e                     # elementary change [C]
epsilon_0 = const.epsilon_0     # vacuum permittivity [F/m]
h = const.h                     # Planck constant [J/Hz]
k_B = const.k                   # Boltzmann constant [J/K]
mu = const.m_u                  # atomic mass unit [kg]


def gamow(m_i, m_k, Z_i, Z_k, E):
    """
    Computing the Gamow peak for two element types i and k at a given temperature T in
    an energy range E.

    m_i = mass of element type i [kg]
    m_k = mass of element type k [kg]
    Z_i = atomic number of element type i
    Z_k = atomic number of element type k
    E = energy interval [J]
    """
    T = 1.57e7 # solar temperature [K]

    m = (m_i*m_k) / (m_i + m_k) # reduced mass

    # exponential functions lambda (proportionality function) and sgima (cross section)
    lmbda = np.exp(-E/(k_B*T))
    sigma = np.exp(-np.sqrt(m/(2*E)) * (Z_i*Z_k*e**2*np.pi) / (epsilon_0*h))

    # finding the peak (maximum value) from the combination of the exponential
    gamow = lmbda*sigma
    peak = gamow[np.where(gamow == np.max(gamow))[0][0]]

    return gamow  / peak # normalized Gamow peak

n = 1000
E = np.logspace(-17, -13, n) # energy interval [J]

# element masses
m_H = 1.007825*mu           # Hydrogen [kg]
m_D2 = 2.014*mu             # Deuterium [kg]
m_He3 = 3.0160*mu           # Helium 3 [kg]
m_He4 = 4.0026*mu           # Helium 4 [kg]
m_Be7 = 7.01693*mu		    # Beryllium 7 [kg]
m_Be8 = 8.00531*mu		    # Beryllium 8 [kg]
m_Li7 = 7.016004*mu	        # Lithium 7 [kg]

m_C12 = 12*mu				# Carbon 12 mass [kg]
m_C13 = 13.00336*mu			# Carbon 13 mass [kg]
m_N13 = 13.00574*mu			# Nitrogen 13 mass [kg]
m_N14 = 14.00307*mu			# Nitrogen 14 mass [kg]
m_N15 = 15.00109*mu			# Nitrogen 15 mass [kg]

# atomic numbers
Z_H = 1     # helium
Z_He = 2    # helium
Z_Be = 3    # beryllium
Z_Li = 3    # lithium

Z_C = 6     # carbon
Z_N = 7     # nitrogen
Z_O = 8     # oxygen

# calculating gamow peaks for all reactions of the PP chain and CNO cycle
gamows = np.zeros((10, n))
PP0_1 = gamow(m_H, m_H, Z_H, Z_H, E); gamows[0] = PP0_1
PP0_2 = gamow(m_D2, m_H, Z_H, Z_H, E); gamows[1] = PP0_2

PP1 = gamow(m_He3, m_He3, Z_He, Z_He, E); gamows[2] = PP1

PP2_1 = gamow(m_He3, m_He4, Z_He, Z_He, E); gamows[3] = PP2_1
PP2_2 = gamow(m_Li7, m_H, Z_Li, Z_H, E); gamows[4] = PP2_2

PP3 = gamow(m_Be7, m_H, Z_Be, Z_H, E); gamows[5] = PP3

CNO_1 = gamow(m_C12, m_H, Z_C, Z_H, E); gamows[6] = CNO_1
CNO_2 = gamow(m_C13, m_H, Z_C, Z_H, E); gamows[7] = CNO_2
CNO_3 = gamow(m_N14, m_H, Z_N, Z_H, E); gamows[8] = CNO_3
CNO_4 = gamow(m_N15, m_H, Z_N, Z_H, E); gamows[9] = CNO_4

plt.figure(figsize=(10, 6))

for i in range(10):
    plt.plot(E, gamows[i, :])

labels = ["pp", "pd", "33", "34", "e7", "17'", "17", "p12", "p13", "p14", "p15"]
plt.xlabel('Energy [J]')
plt.ylabel('Normalized probability')
plt.xscale('log')
plt.legend(labels)
plt.title("Gamow peaks")
plt.savefig("gamow2.png")
plt.show()
