import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import scipy.constants as sc
from tabulate import tabulate
import pandas as pd
from energy_production import energy # project 1

plt.style.use('ggplot')
plt.rcParams['font.size'] = 20
plt.rcParams["lines.linewidth"] = 2

class stellar_modelling:
    """
    This class involves modelling the central part of a Sun-like star,
    including both radiative and convective energy transport.

    The class contains the following functions:
        _readfile() -
        _polation_opacity(T, rho) -
        _pressure(rho, T) - computes pressure in a star for a given density and temperature
        _density(P, T) - computes density in a star for a given pressure and temperature
        _sanity_check_opacity(self)
    """
    def __init__(self, value = int):
        # useful constants
        self.k_B = sc.k                     # Boltzmann constant [J K^-1]
        self.m_u = sc.m_u                   # atomic mass unit [kg]
        self.sigma = sc.sigma               # Stefan-Boltzmann constant [W m^-2 K^-4]
        self.c = sc.c                       # speed of light [m s^-1]
        self.a = 4*self.sigma/self.c        # radiation density constant
        self.G = sc.G                       # constant of gravitaiton [m^3 kg^-1 s^-2]

        # solar parameters from appendix B in lecture notes
        self.L_sun = 3.846e26               # solar luminosisty [W]
        self.M_sun = 1.989e30               # solar mass [kg]
        self.R_sun = 6.96e8                 # solar radius [m]

        # initial parameters given in task description
        self.rho_sun = 1.408e3              # average density of the Sun [kg m^-3]
        self.L_0 = self.L_sun               # luminosity [W]
        self.R_0 = self.R_sun               # solar mass [kg]
        self.M_0 = self.M_sun               # solar radius [m]
        self.rho_0 = 1.42e-7*self.rho_sun   # density [kg m^-3]
        self.T_0 = 5770                     # [K]

        # mass fraction
        self.X = .7                         # hydrogen
        self.Y_He3 = 1e-10                  # helium 3
        self.Y = .29                        # helium 4
        self.Z_Li7 = 10e-7                  # lithium 7
        self.Z_Be7 = 10e-7                  # beryllium 7
        self.Z_N14 = 10e-11                 # nitrogen 14

        # mean molecular weight per particle
        self.mu = 1 / (2*self.X + self.Y_He3 + 3/4*self.Y + 4/7*self.Z_Li7 + 5/7*self.Z_Be7 + 8/14*self.Z_N14)

    def readfile(self):
        """
        This function reads file "opacity.txt" and extracts log(R), log(T) and log(kappa) from the file.
        The .txt file has the following form:
            - Top row is log(R) where R = rho/(T/10^6)^3 and rho is given in [g/cm^3]
            - First column is log(T) with T given in [K]
            - The rest of the table is log(kappa) given in [cm^2/g]
        """
        # loading file
        data = np.loadtxt("/Users/rebeccanguyen/Documents/GitHub/V23/AST3310 Astrophysical Plasma and Stellar interiors/opacity.txt", dtype = np.str)

        # extracting first row and removing first element
        logR = data[0, 1:]
        self.logR = np.asarray(logR, dtype = float)

        # extracting first column
        logT = data[1:, 0];
        self.logT = np.asarray(logT, dtype = float)

        # extracing the remaining matrix
        logkappa = data[1:, 1:]
        self.logkappa = np.asarray(logkappa, dtype = float)

        # interpolation of opacity values (extrapolated for values outside the bounds)
        # calling RectBivariateSpline outputs an array
        self.polation = RectBivariateSpline(self.logT, self.logR, self.logkappa)


    def _polation_opacity(self, T, rho):
        """
        Returns kappa in SI units for a given temperature (T) and density (rho).
        A warning is put out for input values outside the bounds of "opacity.txt".

        Arguments:
            - Temperature, T [K]
            - Density, rho [kg m^-3].
        """
        _rho = np.log10(rho*1000/(T*1e6)**3) # obtaining logR from rho [g/cm^3]

        if T > self.logT[-1] or T < self.logT[0] or _rho < self.logR[0] or _rho > self.logR[-1]:
            print("Warning! Input out of bounds with table.")

        log_kappa = self.polation(T, rho)[0][0]

        return log_kappa

    def _pressure(self, rho, T):
        """
        Computing pressure in a star as a function of density (rho) and temperature (T) given by
        P = P_gas + P_rad
        """
        P_rad = a*T**4/3
        P_gas = rho*self.k_B*T / (self.mu*self.m_u)
        P = P_rad + P_gas
        return P

    def _density(self, P, T):
        """
        Computing density in a star from equation of state for an ideal gas
        """
        P_rad = a*T**4/3
        rho = (P - P_rad)*self.mu*self.m_u / (self.k_B*T)
        return rho

    ########## Differential equations ##########
    def dr(self, rho, r):
        return 1 / (4 * np.pi * r**2 * rho)

    def dP(self, m, r):
        return - (self.G * m) / (4 * np.pi * r**4)

    def dL(self, T, rho):
        return energy(T, rho).energy_production()

    ########## Sanity checks ##########

    def _sanity_check_opacity(self):
        """
        Sanity check of interpolation of opacity values
        """
        logT = [3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 3.770, 3.775, 3.780, 3.795, 3.800]
        logR = [-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95, -5.80, -5.75, -5.70, -5.55, -5.50] # cgs

        # expected values
        logkappa_cgs_expected = [-1.55, -1.51, -1.57, -1.61, -1.67, -1.33, -1.20, -1.02, -1.39, -1.35, -1.31, -1.16, -1.11]
        kappa_SI_expected = [2.84e-3, 3.11e-3, 2.68e-3, 2.46e-3, 2.12e-3, 4.70e-3, 6.25e-3, 9.45e-3, 4.05e-3, 4.43e-3, 4.94e-3, 6.89e-3, 7.69e-3]

        # computed values
        logkappa_cgs_computed = np.array([self.polation(logT[i], logR[i])[0][0] for i in range(len(logT))])
        kappa_SI_computed = 10**logkappa_cgs_computed*.1

        table = pd.DataFrame({"logκ (cgs), exp"      :  logkappa_cgs_expected,
                              "logκ (cgs), com"      :  logkappa_cgs_computed,
                              "κ (SI), exp"          : kappa_SI_expected,
                              "κ (SI), com"          :  kappa_SI_computed})

        print(table)

    def _sanity_check_gradient(self):
        return None

S = stellar_modelling()
S.readfile()
#S._sanity_check_opacity()
