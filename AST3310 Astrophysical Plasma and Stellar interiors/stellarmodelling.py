import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import scipy.constants as sc
from tabulate import tabulate
import pandas as pd
from energy_production import energy # project 1
from cross_section import cross_section

plt.style.use('ggplot')

class stellar_modelling:
    """
    This class involves modelling the central part of a Sun-like star,
    including both radiative and convective energy transport.
    """
    def __init__(self, value = int):
        # mass fraction
        self.X = .7                         # hydrogen
        self.Y_He3 = 1e-10                  # helium 3
        self.Y = .29                        # helium 4
        self.Z_Li7 = 1e-7                  # lithium 7
        self.Z_Be7 = 1e-7                  # beryllium 7
        self.Z_N14 = 1e-11                 # nitrogen 14

        # mean molecular weight per particle
        self.mu = 1 / (2*self.X + self.Y_He3 + 3/4*self.Y + 4/7*self.Z_Li7 + 5/7*self.Z_Be7 + 8/14*self.Z_N14)

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

        # initial parameters
        self.rho_sun = 1.408e3                          # average density of the Sun [kg m^-3]
        self.L_0 = self.L_sun*1.2                       # luminosity [W]
        self.R_0 = self.R_sun*0.9                       # solar radius [kg]
        self.M_0 = self.M_sun                           # solar mass [m]
        self.rho_0 = 1.42e-7*self.rho_sun*29            # density [kg m^-3]
        self.T_0 = 5770*0.5                             # temperature[K]
        self.P_0 = self._P(self.rho_0, self.T_0)        # pressure

        self.alpha = 1
        self.a = 4 * self.sigma / self.c        # radiation density constant
        self.c_P = 5/2 * self.k_B / (self.mu * self.m_u)
        self.eV = sc.eV     # [J]

    def readfile(self):
        """
        This function reads file "opacity.txt" and extracts log(R), log(T) and log(kappa) from the file.
        The .txt file has the following form:
            - Top row is log(R) where R = rho/(T/10^6)^3 and rho is given in [g/cm^3]
            - First column is log(T) with T given in [K]
            - The rest of the table is log(kappa) given in [cm^2/g]
        """
        # loading file
        data_opacity = np.loadtxt("/Users/rebeccanguyen/Documents/GitHub/V23/AST3310 Astrophysical Plasma and Stellar interiors/opacity.txt", dtype = np.str)

        # extracting first row and removing first element
        logR = data_opacity[0, 1:]
        self.logR = np.asarray(logR, dtype = float)

        # extracting first column and removing first element
        logT = data_opacity[1:, 0]
        self.logT = np.asarray(logT, dtype = float)

        # extracing the remaining matrix
        logkappa = data_opacity[1:, 1:]
        self.logkappa = np.asarray(logkappa, dtype = float)

        # interpolation of opacity values (extrapolated for values outside the bounds)
        # calling RectBivariateSpline outputs an array
        self.polation_opacity = RectBivariateSpline(self.logT, self.logR, self.logkappa)

    def _polation_opacity(self, T, rho):
        """
        Returns kappa in SI units for a given temperature (T) and density (rho).
        A warning is put out for input values outside the bounds of "opacity.txt".

        Arguments:
            - Temperature, T [K]
            - Density, rho [kg m^-3].
        """
        _rho = rho * .001 # cgs -> SI units
        logR = np.log10( _rho / (T * 1e-6 )**3 ) # obtaining logR from rho
        logT = np.log10(T)
        log_kappa = self.polation_opacity.ev(logT, logR)

        #if T > self.logT[-1] or T < self.logT[0] or logR < self.logR[0] or logR > self.logR[-1]:
        #    print("Warning! Input out of bounds with table. Proceeding with extrapolation")

        kappa = 10**log_kappa * .1 # return SI units
        return kappa

    def _P(self, rho, T):
        """
        Computing pressure in a star as a function of density (rho) and temperature (T) given by
        P = P_gas + P_rad
        """
        P_rad = (self.a / 3) * T ** 4                   # radiation pressure
        P_gas = (rho*self.k_B*T) / (self.mu*self.m_u)    # gas pressure
        P = P_rad + P_gas
        return P

    def _rho(self, P, T):
        """
        Computing density in a star from equation of state for an ideal gas
        """
        P_rad = self.a * T ** (4/3)
        rho = ( (P - P_rad) * self.mu * self.m_u ) / (self.k_B * T)
        return rho

####################  Temperature gradients ####################
    def _nabla_stable(self, L, T, m, rho, r, kappa):
        """ Stable temperature gradient """
        nabla_stable = (L * 3 * kappa * rho * self._H_P(rho, T, r, m)) / (64 * np.pi * r**2 * self.sigma * T**4)
        return nabla_stable

    def _nabla_ad(self):
        """ Adiabatic temperature gradient """
        return 2/5 # valid for ideal gas

    def _nabla_star(self, rho, T, r, m, L, kappa):
        """ Actual temperature gradient in star """
        xi = self._xi(rho, T, r, m, L, kappa)
        H_P = self._H_P(rho, T, r, m)
        l_m = H_P * self.alpha # mixing length
        K = 4 / l_m ** 2
        U = self._U(rho, r, m, T, kappa)
        nabla_ad = self._nabla_ad()
        nabla_star = xi ** 2 + U * K * xi + nabla_ad

        return nabla_star

    def _nabla_p(self, rho, T, r, m, L, kappa):
        xi = self._xi(rho, T, r, m, L, kappa)
        H_P = self._H_P(rho, T, r, m)
        l_m = H_P * self.alpha # mixing length
        nabla_ad = self._nabla_ad()
        U = self._U(rho, r, m, T, kappa)
        r_p = l_m / 2

        nabla_p = 2 * U * 2 / r_p * xi**2 + nabla_ad
        return nabla_p

####################  Flux ####################
    def _F_rad(self, rho, T, r, m, L, kappa):
        """ Radiative flux """
        nabla_star = self._nabla_star(rho, T, r, m, L, kappa)
        H_P = self._H_P(rho, T, r, m)
        kappa = self._polation_opacity(T, rho)
        F_rad = 16 * self.sigma * T**4 / (3 * kappa * rho * H_P) * nabla_star
        return F_rad

    def _F_con(self, rho, T, r, m, L, kappa):
        """ Convective flux """
        nabla_stable = self._nabla_stable(L, T, m, rho, r, kappa)
        H_P = self._H_P(rho, T, r, m)
        kappa = self._polation_opacity(T, rho)
        F_tot = (16 * self.sigma * T**4) / (3 * kappa * rho * H_P) * nabla_stable
        F_rad = self._F_rad(rho, T, r, m, L, kappa)
        F_con = F_tot - F_rad
        return F_con

##############################################

    def _xi(self, rho, T, r, m, L, kappa):
        """ Calculating the cubic polynomial in order to express _nabla_star() """
        H_P = self._H_P(rho, T, r, m)
        c_P = self.c_P
        U = self._U(rho, r, m, T, kappa)
        nabla_stable = self._nabla_stable(L, T, m, rho, r, kappa)
        nabla_ad = self._nabla_ad()
        l_m = H_P * self.alpha # mixing length

        K = 4 / l_m ** 2
        roots = np.roots(np.array([1, U / l_m ** 2, U * K / l_m ** 2 , (-U / l_m ** 2) * (nabla_stable - nabla_ad)]))
        xi = roots[np.isreal(roots)].real[0]
        return xi

    def _v(self, rho, T, r, m, L, kappa):
        """ Parcel velocity """
        g = self.G * m / r**2
        H_P = self._H_P(rho, T, r, m)
        c_P = self.c_P
        U = self._U(rho, r, m, T, kappa)
        nabla_stable = self._nabla_stable(L, T, m, rho, r, kappa)
        nabla_ad = self._nabla_ad()
        l_m = H_P * self.alpha # mixing length
        xi = self._xi(rho, T, r, m, L, kappa)
        v = np.sqrt( g / H_P ) * l_m/2 * xi
        return v

    def _H_P(self, rho, T, r, m):
        """ Pressure scale height """
        g = self.G * m / r**2
        H_p = self.k_B * T / (self.mu * self.m_u * g)
        return H_p

    def _U(self, rho, r, m, T, kappa):

        c_P = self.c_P
        g = self.G * m / r**2
        H_P = self._H_P(rho, T, r, m)
        U = (64 * self.sigma * T**3) / (3 * kappa * rho**2 * c_P) * np.sqrt( H_P / g )
        return U

####################  Main body ####################
    def _integration(self, m, r, P, L, T, p = 0.01):
        """
        Euler's method to solve the four partial differential
        equations from the stellar surface and to the core.

        Arguments:
        m - mass
        r - radius
        P - pressure
        L - luminosisty
        T - temperature
        """
        rho = self._rho(P, T)
        kappa = self._polation_opacity(T, rho)

        nabla_star = self._nabla_star(rho, T, r, m, L, kappa)
        nabla_stable = self._nabla_stable(L, T, m, rho, r, kappa)
        nabla_ad = self._nabla_ad()
        F_con = self._F_con(rho, T, r, m, L, kappa)
        F_rad = self._F_rad(rho, T, r, m, L, kappa)

        # obtaining epsilon from project 1
        star = energy(T, rho)                            # creating an instance in class energy with given temperature and density
        star.reaction_rates()                            # reaction rates calculated based on given temperature
        PP1, PP2, PP3, CNO, all = star.energy_production()
        eps = all                                        # epsilon = PP0 + PP1 + PP2 + PP3 [J]

        # partial differential equations
        dr = 1 / (4 * np.pi * r**2 * rho)
        dP = - (self.G * m) / (4 * np.pi * r**4)
        dL = eps

        # convetive instability check to determine dT
        if nabla_stable > nabla_ad:
            dT = nabla_star * T / P * dP                                       # convective and radiative transport
            nabla_star = nabla_star
        else:
            dT = - 3 * kappa * L / (256 * np.pi**2 * self.sigma * r**4 * T**3) # radiative transport only
            nabla_star = nabla_stable

        # implementing variable time step
        f = (np.array([np.abs(dr), np.abs(dP), np.abs(dL), np.abs(dT)]))
        V = np.array([r, P, L, T])
        dm = - np.min(p * V / f)

        # new values
        r_new = r + dr * dm
        P_new = P + dP * dm
        L_new = L + dL * dm
        T_new = T + dT * dm
        M_new = m + dm


        return r_new, P_new, L_new, T_new, M_new, rho, nabla_stable, nabla_star, F_con, F_rad, PP1, PP2, PP3, CNO

    def _computation(self):
        """
        This function computes the following values for our star using the function _integration():
        mass, radius, luminosity, convective flux, pressure, density, nabla_stable, nabla_star.

        Output is arrays of these computed values.
        """
        radius = [self.R_0]
        pressure = [self.P_0]
        luminosity = [self.L_0]
        temperature = [self.T_0]
        mass = [self.M_0]
        density = [self.rho_0]
        nabla_stable = []
        nabla_star = []
        F_con = []
        F_rad = []
        PP1_list = []
        PP2_list = []
        PP3_list = []
        CNO_list = []


        i = 0
        while radius[i] > 0 and mass[i] > 0 and luminosity[i] > 0:
            """
            While loop runs until we hit the stellar core, i.e. r = 0
            """
            r_new, P_new, L_new, T_new, M_new, rho_new, nabla_stable_new, nabla_star_new, F_con_new, F_rad_new, PP1, PP2, PP3, CNO = self._integration(mass[i], radius[i], pressure[i], luminosity[i], temperature[i])

            radius.append(r_new)
            pressure.append(P_new)
            luminosity.append(L_new)
            temperature.append(T_new)
            mass.append(M_new)
            density.append(rho_new)
            nabla_stable.append(nabla_stable_new)
            nabla_star.append(nabla_star_new)
            F_con.append(F_con_new)
            F_rad.append(F_con_new)
            PP1_list.append(PP1)
            PP2_list.append(PP2)
            PP3_list.append(PP3)
            CNO_list.append(CNO)

            i += 1

        return np.array(temperature), np.array(mass), np.array(radius), np.array(luminosity), np.array(F_con), np.array(F_rad),\
               np.array(pressure), np.array(density), np.array(nabla_stable), np.array(nabla_star), np.array(PP1_list), np.array(PP2_list), np.array(PP3_list), np.array(CNO_list)

    def _convergence(self):
        """ Check to see if M, R, and L goes to 0 or within 5% of initial values """
        T, M, R, L, F_con, F_rad, P, rho, nabla_stable, nabla_star, PP1, PP2, PP3, CNO = self._computation()

        # removing last element which is negativ
        T = T[:-1]
        M = M[:-1]
        R = R[:-1]
        L = L[:-1]
        P = P[:-1]
        rho = rho[:-1]

        print(f"M/M_0: {M[-1]/self.M_0*100: 4.1f} %")
        print(f"R/R_0: {R[-1]/self.R_0*100: 4.1f} %")
        print(f"L/L_0: {L[-1]/self.L_0*100: 4.1f} %")

#################### Plotting ####################
    def _plotting(self, main_parameters = False, energy_transport = False, energy_production = False, nabla = False):
        """
        A function for plotting the following figures:
        - main parameters
        - energy transport: convective and radiative flux
        - relative energy production
        - temperature gradients
        """

        T, M, R, L, F_con, F_rad, P, rho, nabla_stable, nabla_star, PP1, PP2, PP3, CNO = self._computation()
        eps = PP1 + PP2 + PP3 + CNO

        if main_parameters:
            fig, ax = plt.subplots(3, sharex = True, figsize = (6,5))
            ax[0].plot(R/self.R_0, M/self.M_0, label = r"$M/M_0$")
            ax[0].plot(R/self.R_0, L/self.L_0, label = r"$L/L_0$")
            ax[0].invert_xaxis()
            ax[0].legend()
            ax[0].set_title("Main parameters")

            ax[1].plot(R/self.R_0, T, label = r"$T$")
            ax[1].invert_xaxis()
            ax[1].legend()

            ax[2].plot(R/self.R_0, P/self.P_0, label = r"$P/P_0$")
            ax[2].plot(R/self.R_0, rho/self.rho_0, label = r"$\rho/\rho_0$")
            ax[2].set_yscale("log")
            ax[2].invert_xaxis()
            ax[2].legend()
            ax[2].set_xlabel(r"$R/R_0$")
            plt.savefig("main_parameters_invertx_corrected1.png")

        if energy_transport:
            plt.plot(R[:-1]/self.R_0, F_con , label = r"F$_{con}$")
            plt.plot(R[:-1]/self.R_0, F_rad , label = r"F$_{rad}$")
            plt.legend()
            plt.xlabel(r"$R/R_0$")

        if energy_production:
            fig, ax = plt.subplots(1, sharex = True, figsize = (10, 5))
            ax.plot(R[:-1]/self.R_0, PP1/eps, label = r"$PP1/\varepsilon$")
            ax.plot(R[:-1]/self.R_0, PP2/eps, label = r"$PP2/\varepsilon$")
            ax.plot(R[:-1]/self.R_0, PP3/eps, label = r"$PP3/\varepsilon$")
            ax.plot(R[:-1]/self.R_0, CNO/eps, label = r"$CNO/\varepsilon$")
            plt.plot(R[:-1]/self.R_0, eps/np.max(eps), label = r"$\varepsilon/\varepsilon_{max}$")
            ax.invert_xaxis()
            ax.set_xlabel(r"$R/R_\odot$")
            ax.set_ylabel("Relative energy")
            ax.set_title("Relative energy production from PP chain and CNO cycle")
            ax.legend()
            #plt.savefig("relative_energy_corrected1.png")

        if nabla:
            fig, ax = plt.subplots(1, figsize = (10, 5))
            ax.plot(R[:-1]/self.R_0, nabla_stable, label = r"$\nabla_{stable}$")
            ax.plot(R[:-1]/self.R_0, nabla_star, label = r"$\nabla^*$")
            ax.hlines(self._nabla_ad(), 0, 1, label = r"$\nabla_{ad}$", color = "purple")
            ax.set_yscale("log")
            ax.set_xlabel(r"$R/R_0$")
            ax.set_ylabel(r"$\nabla$")
            ax.set_title("Temperature gradients")
            ax.invert_xaxis()
            ax.legend()
            #plt.savefig("final_temperature_gradients_corrected1.png")


#################### Sanity checks ####################

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
        logkappa_cgs_computed = np.array([self.polation_opacity.ev(logT[i], logR[i]) for i in range(len(logT))])
        kappa_SI_computed = 10**logkappa_cgs_computed*.1

        table = pd.DataFrame({"logκ (cgs), exp"      :  logkappa_cgs_expected,
                              "logκ (cgs), com"      :  logkappa_cgs_computed,
                              "κ (SI), exp"          :  kappa_SI_expected,
                              "κ (SI), com"          :  kappa_SI_computed})

        print(table)

    def _sanity_check_gradient(self):
        """ Checking our functions up against example 5.1 """
        # defining test parameters
        mu = self.mu
        T = .9e6                # [K]
        rho = 55.9              # [kg m^-3]
        R = .84*self.R_sun
        M = .99*self.M_sun
        kappa = 3.98            # [m^2 kg^-1]
        alpha = 1
        L = self.L_0            # I think we are supposed to use this

        # expected values
        nabla_stable_exp = 3.26
        nabla_ad_exp = 2/5
        H_P_exp = 32.4              # [Mm]
        U_exp = 5.94e5              # [m^2]
        xi_exp = 1.173e-3
        nabla_star_exp = .4
        v_exp = 65.6                # [m s^-1]
        F_con_exp = .88             # F_con / (F_con + F_rad)
        F_rad_exp = .12             # F_rad / (F_con + F_rad)

        # computing values
        nabla_stable_com = self._nabla_stable(L, T, M, rho, R, kappa)
        nabla_ad_com = self._nabla_ad()
        nabla_star_com = self._nabla_star(rho, T, R, M, L, kappa)
        H_P_com = self._H_P(rho, T, R, M) / 1e6  # [Mm]
        U_com = self._U(rho, R, M, T, kappa)
        xi_com = self._xi(rho, T, R, M, L, kappa)
        v_com = self._v(rho, T, R, M, L, kappa)
        F_con_com = self._F_con(rho, T, R, M, L, kappa) / (self._F_con(rho, T, R, M, L, kappa)  + self._F_rad(rho, T, R, M, L, kappa))
        F_rad_com = self._F_rad(rho, T, R, M, L, kappa) / (self._F_con(rho, T, R, M, L, kappa)  + self._F_rad(rho, T, R, M, L, kappa))

        expected = ["Expected", nabla_stable_exp, nabla_ad_exp, nabla_star_exp, H_P_exp, U_exp, xi_exp, v_exp, F_con_exp, F_rad_exp]
        computed = ["Computed", nabla_stable_com, nabla_ad_com, nabla_star_com, H_P_com, U_com, xi_com, v_com, F_con_com, F_rad_com]
        print(tabulate([expected, computed], headers = ["∇_stable", "∇_ad", "∇*", "H_P", "U", "ξ", "v", "F_con", "F_rad"]))

        nabla_p = self._nabla_p(rho, T, R, M, L, kappa)
        criterion = nabla_stable_com > nabla_star_com and nabla_star_com > nabla_p and nabla_p > nabla_ad_com
        print(f"∇_stable > ∇* > ∇_p > ∇_ad satisfied?: {criterion}")
        return None

    def _sanity_check_temperatures_gradient_plot(self):
        T, M, R, L, F_con, F_rad, P, rho, nabla_stable, nabla_star, PP1, PP2, PP3, CNO = self._computation()
        R = R[:-1]
        plt.figure(figsize = (6, 4))
        plt.plot(R/self.R_sun, nabla_stable, label = r"$\nabla_{stable}$")
        plt.plot(R/self.R_sun, nabla_star, label = r"$\nabla^*$")
        plt.hlines(self._nabla_ad(), 0, 1, label = r"$\nabla_{ad}$", color = "purple")
        plt.xlabel(r"$R/R_0$")
        plt.ylabel(r"$\nabla$")
        plt.yscale("log")
        plt.legend()
        plt.title("Temperature gradients (for p = 0.01)")
        plt.savefig("sanity_temperature_gradients2.png")


if __name__ == "__main__":
    S = stellar_modelling()
    S.readfile()

    """Sanity checks"""
    #S._sanity_check_opacity()
    #S._sanity_check_gradient()
    #S._sanity_check_temperatures_gradient_plot()

    """Convergence test"""
    #S._convergence()

    """Cross section"""
    #T, M, R, L, F_con, F_rad, P, rho, nabla_stable, nabla_star, PP1, PP2, PP3, CNO, eps = S._computation()
    #cross_section(R, L, F_con, show_every = 50, sanity = False, savefig = False)

    """Plotting"""
    S._plotting(main_parameters = True, energy_transport = False, energy_production = False, nabla = False)

    plt.show()
