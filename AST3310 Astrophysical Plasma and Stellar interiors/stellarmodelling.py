import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

plt.style.use('ggplot')
plt.rcParams['font.size'] = 20
plt.rcParams["lines.linewidth"] = 2

class stellar_modelling:
    """
    This class involves modelling the central part of a Sun-like star,
    including both radiative and convective energy transport.

    The class contains the following functions:
        readfile() -
    """
    def __init__(self):
        # useful constants

        # solar parameters from appendix B in lecture notes
        self.L_sun = 3.846e26    # solar luminosisty [W]
        self.M_sun = 1.989e30    # solar mass [kg]
        self.R_sun = 6.96e8      # solar radius [m]

        # initial parameters given in task description
        self.rho_sun = 1.408e3   # average density of the Sun [kg m^-3]
        self.L_0 = self.L_sun
        self.R_0 = self.R_sun
        self.M_0 = self.M_sun
        self.rho_0 = 1.42e-7*self.rho_sun
        self.T_0 = 5770          # [K]

        # mass fraction
        self.X = .7              # hydrogen
        self.Y_He3 = 1e-10       # helium 3
        self.Y = .29             # helium 4
        self.Z_Li7 = 10e-7       # lithium 7
        self.Z_Be7 = 10e-7       # beryllium 7
        self.Z_N14 = 10e-11      # nitrogen 14

        self.logR = None
        self.logT = None
        self.logkappa = None
    def readfile(self):
        """
        This function reads file "opacity.txt" and extracts log(R), log(T) and log(kappa) from the file.

        The file has the following form:
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

    def interpolate_extrapolate(self, T, rho):
        """
        Input values are temperature, T, given in [K] and density, rho, given in [kg m^-3].
        """
        self.readfile() # read "opacity.txt"

        interpolate = RectBivariateSpline(self.logT, self.logR, self.logkappa)
        rho_to_R = np.log10(rho*1000/(T*1e6)**3) # obtaining logR from rho [g/cm^3]

        if T > self.logT[-1] or T < self.logT[0] or rho_to_R < self.logR[0] or rho_to_R > self.logR[-1]:
            print("Warning! Proceeding with extrapolation.")

stellar_modelling().readfile()
stellar_modelling().interpolate_extrapolate(-7, 1)
