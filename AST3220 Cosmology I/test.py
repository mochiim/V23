import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
import astropy.constants as const
from scipy import interpolate

class ElementAbundance:

    """Simulate the reaction rates and element synthesis in the BBN."""

    def __init__(self):
        self.c = const.c.cgs.value         # light speed
        self.k_B = const.k_B.cgs.value     # Boltzmann constant
        self.G = const.G.cgs.value         # Newton's gravitational constant
        self.h_bar = const.hbar.cgs.value  # reduced Planck constant
        self.m_e = const.m_e.cgs.value     # electron mass
        self.m_p = const.m_p.cgs.value     # proton mass
        self.m_n = const.m_n.cgs.value     # neutron mass

        self.Mpc = 3.086e24         # [cm] Megaparsec
        self.h = 0.7                # hubble constant
        self.N_eff = 3              # neutrino species
        self.T0 = 2.725             # [K] today's temperature of the CMB
        self.H0 = 100 * self.h * 1e5 / self.Mpc # Hubble parameter in CGS
        self.tau = 1700             # [s] free neutron decay time

        self.rho_c0 = 9.2e-27 * 1000/100**3
        # self.rho_c0 = 9.2e-27 * 1e3 / 1e5 # This lead to a major numerical error!

        self.omega_b0 = 0.05

    def dY_f(self, logT, Y):
        """
        Differential eqs for dYn/dlnT, dYp/dlnT from eqs (10), (11).
        Work out dY/dlnT for the first two weak reactions in table 2.
        """
        T = np.exp(logT) # [K] temperature
        a = self.T0 / T

        omega_r0 = (8 * np.pi**3 * self.G) / (90 * self.H0**2) \
                    * (self.k_B * self.T0)**4 / (self.h_bar**3 * self.c**5) \
                    * (2 + self.N_eff * 7/4 * (4/11)**(4/3))

        H = self.H0 * np.sqrt(omega_r0) * a**(-2)
        Y_n = Y[0]; Y_p = Y[1]

        lambda_n = self.gamma_n_p(T)
        lambda_p = self.gamma_p_n(T)

        dY_n = -1 / H * (Y_p * lambda_p - Y_n * lambda_n )
        dY_p = -1 / H * (Y_n * lambda_n - Y_p * lambda_p )

        return np.array([dY_n, dY_p])

    def task_f(self):
        """Solve differential equations for dYn and dYp and plot the results.
        """
        # inial conditions for Y_n  and Y_p:
        T_i = 1e11
        Yn_Ti = (1 + np.exp( (self.m_n - self.m_p) \
                            * self.c**2 / self.k_B * T_i ) )**(-1)
        Yp_Ti = 1 - Yn_Ti

        solve = int.solve_ivp(self.dY_f, t_span=[np.log(100e9), np.log(0.1e9)],\
                y0=[Yn_Ti, Yp_Ti], method='Radau', rtol=1e-12, atol=1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]
        Y_p = solve.y[1]

        plt.plot(T, Y_p, label='p')
        plt.plot(T, Y_n, label='n')
        plt.plot(T, 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T))),\
                 linestyle="dotted", color="tab:orange")
        plt.plot(T, 1 - 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T))), \
                 linestyle="dotted", color="tab:blue")
        plt.xscale('log'); plt.yscale('log')
        plt.xlim([T[0], T[-1]]); plt.ylim([1e-3, 2])
        plt.xlabel('$T$ [K]'); plt.ylabel(r'$Y_i$')
        plt.legend()
        # plt.savefig('f.png')
        plt.show()

    def dY_h(self, logT, Y):
        """Differential eqs from (20), (21) and (22).
        """
        T = np.exp(logT)
        a = self.T0 / T
        T9 = T / 1e9

        omega_r0 = (8 * np.pi**3 * self.G) / (90 * self.H0**2) \
                    * (self.k_B * self.T0)**4 / (self.h_bar**3 * self.c**5) \
                    * (2 + self.N_eff * (7/4) * (4/11)**(4/3))

        H = self.H0 * np.sqrt(omega_r0) * a**(-2)
        Y_n = Y[0]; Y_p = Y[1]; Y_D = Y[2]

        lambda_n = self.gamma_n_p(T)
        lambda_p = self.gamma_p_n(T)
        p_n, lambda_D = self.pn_to_Dgamma(T)

        dY_n = -1/H * (- lambda_n*Y_n + lambda_p*Y_p + lambda_D*Y_D \
                         - p_n * Y_n * Y_p)
        dY_p = -1/H * (- lambda_p*Y_p + lambda_n*Y_n + lambda_D*Y_D \
                         - p_n * Y_n * Y_p)
        dY_D = -1/H *(- lambda_D*Y_D + p_n * Y_n * Y_p)

        return np.array([dY_n, dY_p, dY_D])

    def task_h(self):
        """Solve task f and plot results.
        """
        # inial conditions for Y_n, Y_p and Y_D:
        T_i = 1e11
        Yn_Ti = (1 + np.exp( (self.m_n - self.m_p) \
                            * self.c**2 / self.k_B * T_i ) )**(-1)
        Yp_Ti = 1 - Yn_Ti
        YD = 0

        solve = int.solve_ivp(self.dY_h, t_span=[np.log(1e11), np.log(0.1e9)],\
                y0=[Yn_Ti, Yp_Ti, YD], method='Radau', rtol=1e-12, atol=1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]
        Y_p = solve.y[1]
        Y_D = solve.y[2]

        plt.plot(T, Y_p, label='p')
        plt.plot(T, Y_n, label='n')
        plt.plot(T, 2*Y_D, label='D')
        plt.plot(T, 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T))),\
                 linestyle="dotted", color="tab:orange")
        plt.plot(T, 1 - 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T))), \
                 linestyle="dotted", color="tab:blue")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('$T$ [K]'); plt.ylabel(r'Mass fraction $A_i Y_i$')
        plt.xlim([T[0], T[-1]])
        plt.ylim([1e-3, 2])
        plt.legend()
        plt.tight_layout()
        # plt.savefig('h.png')
        plt.show()

    def dY_i(self, logT, Y):
        T = np.exp(logT)
        a = self.T0 / T
        T9 = T / 1e9

        omega_r0 = (8 * np.pi**3 * self.G) / (90 * self.H0**2) \
                    * (self.k_B * self.T0)**4 / (self.h_bar**3 * self.c**5) \
                    * (2 + self.N_eff * (7/4) * (4/11)**(4/3))

        H = self.H0 * np.sqrt(omega_r0) * a**(-2)

        Y_n = Y[0]; Y_p = Y[1]; Y_D = Y[2]; Y_T = Y[3]
        Y_He3 = Y[4]; Y_He4 = Y[5]; Y_Li7 = Y[6]; Y_Be7 = Y[7]

        dY_n, dY_p, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7 \
            = 0, 0, 0, 0, 0, 0, 0, 0

        # Weak)
        dY_n = self.dY_f(logT, Y)[0]
        dY_p = self.dY_f(logT, Y)[1]

        # 1)
        dY_n = self.dY_h(logT, Y)[0]
        dY_p = self.dY_h(logT, Y)[1]
        dY_D = self.dY_h(logT, Y)[2]

        # 2)
        p_D, lambda_He3 = self.pD_to_He3gamma(T)
        dY_p = dY_p     - 1/H * (-Y_p * Y_D * p_D + Y_He3 * lambda_He3)
        dY_D = dY_D     - 1/H * (-Y_p * Y_D * p_D + Y_He3 * lambda_He3)
        dY_He3 = dY_He3 - 1/H * (Y_p * Y_D * p_D - Y_He3 * lambda_He3)

        # 3)
        n_D, lambda_T = self.nD_to_Tgamma(T)
        dY_n = dY_n - 1/H * (Y_D * Y_T * lambda_T - Y_n * Y_D * n_D)
        dY_D = dY_D - 1/H * (Y_D * Y_T * lambda_T - Y_n * Y_D * n_D)
        dY_T = dY_T - 1/H * (Y_n * Y_D * n_D - Y_T * lambda_T)

        # 4)
        n_He3, lambda_pT = self.nHe3_to_pT(T)
        dY_n = dY_n     - 1/H * (Y_p * Y_T * lambda_pT - Y_n * Y_He3 * n_He3)
        dY_He3 = dY_He3 - 1/H * (Y_p * Y_T * lambda_pT - Y_n * Y_He3 * n_He3)
        dY_p = dY_p     - 1/H * (Y_n * Y_He3 * n_He3 - Y_p * Y_T * lambda_pT)
        dY_T = dY_T     - 1/H * (Y_n * Y_He3 * n_He3 - Y_p * Y_T * lambda_pT)

        # 5)
        p_T, lambda_He4 = self.pT_to_He4gamma(T)
        dY_p = dY_p     - 1/H * (Y_He4 * lambda_He4 - Y_p * Y_T * p_T)
        dY_T = dY_T     - 1/H * (Y_He4 * lambda_He4 - Y_p * Y_T * p_T)
        dY_He4 = dY_He4 - 1/H * (Y_p * Y_T * p_T - Y_He4 * lambda_He4)

        # 6)
        n_He3, lambda_He4 = self.nHe3_to_He4gamma(T)
        dY_n = dY_n     - 1/H * (Y_He4 * lambda_He4 - Y_n * Y_He3 * n_He3)
        dY_He3 = dY_He3 - 1/H * (Y_He4 * lambda_He4 - Y_n * Y_He3 * n_He3)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * lambda_He4 - Y_n * Y_He3 * n_He3)

        # 7)
        D_D, lambda_nHe3 = self.DD_to_nHe3(T)
        dY_D = dY_D     - 1/H * (Y_n * Y_He3 * lambda_nHe3 - Y_D * Y_D * D_D)
        dY_n = dY_n     + 1/H * (Y_n * Y_He3 * lambda_nHe3 - 0.5 * Y_D**2 * D_D)
        dY_He3 = dY_He3 + 1/H * (Y_n * Y_He3 * lambda_nHe3 - 0.5 * Y_D**2 * D_D)

        # 8)
        D_D, lambda_pT = self.DD_to_pT(T)
        dY_D = dY_D - 1/H * (Y_p * Y_T * lambda_pT - Y_D**2 * D_D)
        dY_p = dY_p + 1/H * (Y_p * Y_T * lambda_pT - 0.5 * Y_D**2 * D_D)
        dY_T = dY_T + 1/H * (Y_p * Y_T * lambda_pT - 0.5 * Y_D**2 * D_D)

        # 9)
        D_D, lambda_He4 = self.DD_to_He4gamma(T)
        dY_D = dY_D     - 1/H * (Y_He4 * lambda_He4 - Y_D**2 * D_D)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * lambda_He4 - 0.5 * Y_D**2 * D_D)

        # 10)
        D_He3, lambda_He4p = self.DHe3_to_He4p(T)
        dY_D = dY_D     - 1/H * (Y_He4 * Y_p * lambda_He4p - Y_D * Y_He3 * D_He3)
        dY_He3 = dY_He3 - 1/H * (Y_He4 * Y_p * lambda_He4p - Y_D * Y_He3 * D_He3)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * Y_p * lambda_He4p - Y_D * Y_He3 * D_He3)
        dY_p = dY_p     + 1/H * (Y_He4 * Y_p * lambda_He4p - Y_D * Y_He3 * D_He3)

        # 11)
        D_T, lambda_He4n = self.DT_to_He4n(T)
        dY_D = dY_D     - 1/H * (Y_He4 * Y_n * lambda_He4n - Y_D * Y_T * D_T)
        dY_T = dY_T     - 1/H * (Y_He4 * Y_n * lambda_He4n - Y_D * Y_T * D_T)
        dY_He4 = dY_He4 - 1/H * -(Y_He4 * Y_n * lambda_He4n - Y_D * Y_T * D_T)
        dY_n = dY_n     - 1/H * -(Y_He4 * Y_n * lambda_He4n - Y_D * Y_T * D_T)

        # 15)
        He3_T, lambda_He4D = self.He3T_to_He4D(T)
        dY_He3 = dY_He3 - 1/H * (Y_He4 * Y_D * lambda_He4D - Y_He3 * Y_T * He3_T)
        dY_T = dY_T     - 1/H * (Y_He4 * Y_D * lambda_He4D - Y_He3 * Y_T * He3_T)
        dY_D = dY_D     + 1/H * (Y_He4 * Y_D * lambda_He4D - Y_He3 * Y_T * He3_T)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * Y_D * lambda_He4D - Y_He3 * Y_T * He3_T)

        # 16)
        He3_He4, lambda_Be7 = self.He3He4_to_Be7gamma(T)
        dY_He3 = dY_He3 - 1/H * (Y_Be7 * lambda_Be7 - Y_He3 * Y_He4 * He3_He4)
        dY_He4 = dY_He4 - 1/H * (Y_Be7 * lambda_Be7 - Y_He3 * Y_He4 * He3_He4)
        dY_Be7 = dY_Be7 + 1/H * (Y_Be7 * lambda_Be7 - Y_He3 * Y_He4 * He3_He4)

        # 17)
        T_He4, lambda_Li7 = self.THe4_to_Li7gamma(T)
        dY_T = dY_T     - 1/H * (Y_Li7 * lambda_Li7 - Y_T * Y_He4 * T_He4)
        dY_He4 = dY_He4 - 1/H * (Y_Li7 * lambda_Li7 - Y_T * Y_He4 * T_He4)
        dY_Li7 = dY_Li7 + 1/H * (Y_Li7 * lambda_Li7 - Y_T * Y_He4 * T_He4)

        # 18)
        n_Be7, lambda_pLi7 = self.nBe7_to_pLi7(T)
        dY_n = dY_n     - 1/H * (Y_p * Y_Li7 * lambda_pLi7 - Y_n * Y_Be7 * n_Be7)
        dY_Be7 = dY_Be7 - 1/H * (Y_p * Y_Li7 * lambda_pLi7 - Y_n * Y_Be7 * n_Be7)
        dY_p = dY_p     + 1/H * (Y_p * Y_Li7 * lambda_pLi7 - Y_n * Y_Be7 * n_Be7)
        dY_Li7 = dY_Li7 + 1/H * (Y_p * Y_Li7 * lambda_pLi7 - Y_n * Y_Be7 * n_Be7)

        # 20)
        p_Li7, lambda_2He4 = self.pLi7_to_He4He4(T)
        dY_p = dY_p     - 1/H * (0.5 * Y_He4**2 * lambda_2He4 - Y_p * Y_Li7 * p_Li7)
        dY_Li7 = dY_Li7 - 1/H * (0.5 * Y_He4**2 * lambda_2He4 - Y_p * Y_Li7 * p_Li7)
        dY_He4 = dY_He4 + 1/H * (Y_He4**2 * lambda_2He4 - Y_p * Y_Li7 * p_Li7)

        # 21)
        n_Be7, lambda_2He4 = self.nBe7_to_He4He4(T)
        dY_n = dY_n     - 1/H * (0.5 * Y_He4**2 * lambda_2He4 - Y_n * Y_Be7 * n_Be7)
        dY_Be7 = dY_Be7 - 1/H * (0.5 * Y_He4**2 * lambda_2He4 - Y_n * Y_Be7 * n_Be7)
        dY_He4 = dY_He4 + 1/H * (Y_He4**2 * lambda_2He4 - Y_n * Y_Be7 * n_Be7)

        return np.array([dY_n, dY_p, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7])

    def task_i(self):
        # Initial conditions from e)
        T_i = 100e9
        Yn_Ti = (1 + np.exp( (self.m_n - self.m_p) \
                            * self.c**2 / self.k_B * T_i ) )**(-1)
        Yp_Ti = 1 - Yn_Ti
        YD, YT, YHe3, YHe4, YLi7, YBe7 = 0, 0, 0, 0, 0, 0

        solve = int.solve_ivp(self.dY_i, t_span=[np.log(T_i), np.log(0.01e9)],\
                y0=[Yn_Ti, Yp_Ti, YD, YT, YHe3, YHe4, YLi7, YBe7], \
                method='Radau', rtol=1e-12, atol=1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]
        Y_p = solve.y[1]
        Y_D = solve.y[2]
        Y_T = solve.y[3]
        Y_He3 = solve.y[4]
        Y_He4 = solve.y[5]
        Y_Li7 = solve.y[6]
        Y_Be7 = solve.y[7]

        plt.plot(T, Y_p, label='p')
        plt.plot(T, Y_n, label='n')
        plt.plot(T, 2*Y_D, label='D')
        plt.plot(T, 3*Y_T, label='T')
        plt.plot(T, 3*Y_He3, label=r'He$^3$')
        plt.plot(T, 4*Y_He4, label=r'He$^4$')
        plt.plot(T, 7*Y_Li7, label=r'Li$^7$')
        plt.plot(T, 7*Y_Be7, label=r'Be$^7$')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('$T$ [K]'); plt.ylabel(r'Mass fraction $A_i Y_i$')
        plt.xlim([T[0], T[-1]])
        plt.ylim([1e-11, 1e1])
        plt.legend()
        plt.tight_layout()
        # plt.savefig('i.png')
        plt.show()

        return

    def task_j(self, N):

        # Data (min, max, without error, error)
        dat_YD_Yp = ((2.57 - 0.03)*1e-5, (2.57 + 0.03)*1e-5, 2.57*1e-5, 0.03*1e-5)
        dat_Y_He4 = (0.254 - 0.003, 0.254 + 0.003, 0.254, 0.003)
        dat_Y_Li7_Y_p = ((1.6 - 0.3)*1e-10, (1.6 +0.3)*1e-10, 1.6*1e-10, 0.3*1e-10)

        omegas = np.logspace(-2, 0, N)

        xi = np.zeros(N)

        Y_D = np.zeros(N)
        Y_n, T, Y_p, Y_He3, Y_He4, Y_T, Y_Li7, Y_Be7 \
            = Y_D.copy(), Y_D.copy(), Y_D.copy(), Y_D.copy(), Y_D.copy(), \
              Y_D.copy(), Y_D.copy(), Y_D.copy()

        # Calculating relic abundances:
        for i, val in enumerate(omegas):
            self.omega_b0 = val
            T_i = 100e9
            Yn_Ti = (1 + np.exp( (self.m_n - self.m_p) \
                                * self.c**2 / self.k_B * T_i ) )**(-1)
            Yp_Ti = 1 - Yn_Ti
            YD, YT, YHe3, YHe4, YLi7, YBe7 = 0, 0, 0, 0, 0, 0

            solve = int.solve_ivp(self.dY_i, t_span=[np.log(T_i), np.log(0.01e9)],\
                    y0=[Yn_Ti, Yp_Ti, YD, YT, YHe3, YHe4, YLi7, YBe7], \
                    method='Radau', rtol=1e-12, atol=1e-12)

            T[i]     = np.exp(solve.t[0])
            Y_n[i]   = np.max([solve.y[0][-1], 1e-20])
            Y_p[i]   = np.max([solve.y[1][-1], 1e-20])
            Y_D[i]   = np.max([solve.y[2][-1], 1e-20])
            Y_T[i]   = np.max([solve.y[3][-1], 1e-20])
            Y_He3[i] = np.max([solve.y[4][-1], 1e-20])
            Y_He4[i] = np.max([solve.y[5][-1], 1e-20])
            Y_Li7[i] = np.max([solve.y[6][-1], 1e-20])
            Y_Be7[i] = np.max([solve.y[7][-1], 1e-20])

        int_YD_Yp = interpolate.interp1d(np.log(omegas), np.log(Y_D / Y_p), kind='cubic')
        int_YHe4= interpolate.interp1d(np.log(omegas), np.log(4 * Y_He4), kind='cubic')
        int_YLi7_Yp = interpolate.interp1d(np.log(omegas), np.log((Y_Li7 + Y_Be7) / Y_p), kind='cubic')
        int_YHe3= interpolate.interp1d(np.log(omegas), np.log(Y_He3 + Y_T), kind='cubic')

        xnew = np.logspace(-2, 0, 1000)
        plot_YD_Yp   = np.exp(int_YD_Yp(np.log(xnew)))
        plot_YHe4    = np.exp(int_YHe4(np.log(xnew)))
        plot_YLi7_Yp = np.exp(int_YLi7_Yp(np.log(xnew)))
        plot_YHe3    = np.exp(int_YHe3(np.log(xnew)))

        # Calculating xi-squared:
        xi = (plot_YD_Yp - dat_YD_Yp[2])**2 / dat_YD_Yp[3]**2 \
             + (plot_YHe4 - dat_Y_He4[2])**2 / dat_Y_He4[3]**2 \
             + ( plot_YLi7_Yp - dat_Y_Li7_Y_p[2] )**2 \
             / dat_Y_Li7_Y_p[3]**2

        self.omega_b0 = 0.05 # just in case

        prob = np.exp(-xi) / np.max(np.exp(-xi))
        xi_min = np.argmax(prob)

        # Plotting'
        fig, ax = plt.subplots(3, sharex=True, squeeze=True, \
                               gridspec_kw={'height_ratios': [1, 3, 1]})

        # Data
        ax[0].axhspan(dat_Y_He4[0], dat_Y_He4[1], alpha=0.3, color='tab:green')
        #ax[1].axhspan(dat_YD_Yp[0], dat_YD_Yp[1], alpha=0.3, color='tab:blue')
        #ax[1].axhspan(dat_Y_Li7_Y_p[0], dat_Y_Li7_Y_p[1], alpha=0.3, color='tab:red')

        # Calculated
        ax[0].plot(xnew, plot_YHe4, color='tab:green', label=r'He$^4$')
        #ax[0].set_ylabel(r'4Y$_{He^4}$')
        #ax[0].legend(loc='upper left')
        ax[0].set_ylim([0.20, 0.30]); ax[0].set_xlim([1e-2, 1e0])
        #ax[0].vlines(xnew[xi_min], 0.20, 0.30, color='k',linestyles='dotted')

        #ax[1].plot(xnew, plot_YD_Yp, color='tab:blue', label=r'D')
        #ax[1].plot(xnew, plot_YHe3, color='tab:orange', label=r'He$^3$')
        #ax[1].plot(xnew, plot_YLi7_Yp, color='tab:red', label=r'Li$^7$')
        #ax[1].set_ylim([1e-10, 1e-3]); ax[1].set_xlim([1e-2, 1e0])
        #ax[1].set_ylabel(r'Y$_i$/Y$_p$')
        #ax[1].legend(loc='center left')
        #x[1].vlines(xnew[xi_min], 1e-11, 1e-3, color='k', linestyles='dotted')

        # Normalized probability:
        #ax[2].plot(xnew, prob, color='k')
        #ax[2].set_ylabel(r'Normalized probability')
        #ax[2].set_ylim([0.0, 1.0]); ax[2].set_xlim([1e-2, 1e0])

        # Setting logscale
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        #ax[1].set_xscale("log"); ax[1].set_yscale("log")
        #ax[2].set_xscale("log")
        #ax[2].set_xlabel(r'$\Omega_{b0}$')
        fig.tight_layout()
        # plt.savefig('j.png')
        plt.show()

    def task_k(self, N):

        # Data (min, max, data, error)
        dat_YD_Yp = ((2.57 - 0.03)*1e-5, (2.57 + 0.03)*1e-5, 2.57*1e-5, 0.03*1e-5)
        dat_Y_He4 = (0.254 - 0.003, 0.254 + 0.003, 0.254, 0.003)
        dat_Y_Li7_Y_p = ((1.6 - 0.3)*1e-10, (1.6 +0.3)*1e-10, 1.6*1e-10, 0.3*1e-10)

        Neffs = np.linspace(1, 5, N)

        xi = np.zeros(N)

        Y_D = np.zeros(N)
        Y_n, Y_T, Y_p, Y_He3, Y_He4, Y_T, Y_Li7, Y_Be7 \
            = Y_D.copy(), Y_D.copy(), Y_D.copy(), Y_D.copy(), Y_D.copy(), \
              Y_D.copy(), Y_D.copy(), Y_D.copy()

        # Calculating relic abundances:
        for i, val in enumerate(Neffs):
            self.N_eff = val
            T_i = 100e9
            Yn_Ti = (1 + np.exp( (self.m_n - self.m_p) \
                                * self.c**2 / self.k_B * T_i ) )**(-1)
            Yp_Ti = 1 - Yn_Ti
            YD, YT, YHe3, YHe4, YLi7, YBe7 = 0, 0, 0, 0, 0, 0

            solve = int.solve_ivp(self.dY_i, t_span=[np.log(T_i), np.log(0.01e9)],\
                    y0=[Yn_Ti, Yp_Ti, YD, YT, YHe3, YHe4, YLi7, YBe7], \
                    method='Radau', rtol=1e-12, atol=1e-12)

            T[i]     = np.exp(solve.t[0])
            Y_n[i]   = solve.y[0][-1]
            Y_p[i]   = solve.y[1][-1]
            Y_D[i]   = solve.y[2][-1]
            Y_T[i]   = solve.y[3][-1]
            Y_He3[i] = solve.y[4][-1]
            Y_He4[i] = solve.y[5][-1]
            Y_Li7[i] = solve.y[6][-1]
            Y_Be7[i] = solve.y[7][-1]

        self.N_eff = 3 # Just in case

        int_YD_Yp = interpolate.interp1d(Neffs, Y_D / Y_p, kind='cubic')
        int_YHe4= interpolate.interp1d(Neffs, 4 * Y_He4, kind='cubic')
        int_YLi7_Yp = interpolate.interp1d(Neffs, (Y_Li7 + Y_Be7) / Y_p, kind='cubic')
        int_YHe3= interpolate.interp1d(Neffs, Y_He3 + Y_T, kind='cubic')

        xnew = np.linspace(1, 5, 1000)
        plot_YD_Yp   = int_YD_Yp(xnew)
        plot_YHe4    = int_YHe4(xnew)
        plot_YLi7_Yp = int_YLi7_Yp(xnew)
        plot_YHe3    = int_YHe3(xnew)

        # Calculating xi-squared:
        xi = (plot_YD_Yp - dat_YD_Yp[2])**2 / dat_YD_Yp[3]**2 \
             + (plot_YHe4 - dat_Y_He4[2])**2 / dat_Y_He4[3]**2 \
             + ( plot_YLi7_Yp - dat_Y_Li7_Y_p[2] )**2 \
             / dat_Y_Li7_Y_p[3]**2

        prob = np.exp(-xi) / np.max(np.exp(-xi))
        xi_min = np.argmax(prob)

        # Plotting
        fig, ax = plt.subplots(4, sharex=True, figsize=(6,6))

        # Data
        ax[0].axhspan(dat_Y_He4[0], dat_Y_He4[1], alpha=0.3, color='tab:green')
        ax[1].axhspan(dat_YD_Yp[0], dat_YD_Yp[1], alpha=0.3, color='tab:blue')
        ax[2].axhspan(dat_Y_Li7_Y_p[0], dat_Y_Li7_Y_p[1], alpha=0.3, color='tab:red')

        # Calculated
        ax[0].plot(xnew, plot_YHe4, color='tab:green', label=r'He$^4$')
        ax[0].set_ylabel(r'4Y$_{He^4}$')
        ax[0].legend(loc='upper left')
        ax[0].set_ylim([0.20, 0.30]); ax[0].set_xlim([1.0, 5.0])
        ax[0].vlines(xnew[xi_min], 0.20, 0.40, color='k',linestyles='dotted')

        ax[1].plot(xnew, plot_YD_Yp, color='tab:blue', label=r'D')
        ax[1].plot(xnew, plot_YHe3, color='tab:orange', label=r'He$^3$')
        ax[1].legend(loc='upper left')
        ax[1].set_ylabel(r'Y$_i$/Y$_p$')
        ax[1].set_ylim([0.5e-5, 4e-5]); ax[1].set_xlim([1.0, 5.0])
        ax[1].vlines(xnew[xi_min], 0.5e-5, 4e-5, color='k', linestyles='dotted')

        ax[2].plot(xnew, plot_YLi7_Yp, color='tab:red', label=r'Li$^7$')
        ax[2].set_ylim([1e-10, 3e-10]); ax[2].set_xlim([1.0, 5.0])
        ax[2].legend(loc='upper left')
        ax[2].set_ylabel(r'Y$_i$/Y$_p$')
        ax[2].vlines(xnew[xi_min], 1e-10, 3e-10, color='k', linestyles='dotted')

        # Normalized probability:
        ax[3].plot(xnew, prob, color='k')
        ax[3].set_ylabel(r'Normalized probability')
        ax[3].set_ylim([0.0, 1.0]); ax[3].set_xlim([1.0, 5.0])
        ax[3].set_xlabel(r'$N_{eff}$')
        fig.tight_layout()
        plt.show()



    """Reaction rates from table 2"""

    def gamma_n_p(self, T, q=2.53):
        """Calculate Gamma n to p from eq (12)."""

        T_nu = (4 / 11)**(1 / 3) * T # From b)
        Z_nu = self.m_e * self.c**2 / (self.k_B * T_nu)
        Z = self.m_e * self.c**2 / (self.k_B * T)
        tau = self.tau

        f = lambda x: ((x + q)**2 * (x**2 - 1)**(1/2) * x) \
            / ((1 + np.exp(x * Z)) * (1 + np.exp( -(x + q) * Z_nu ))) \
            + ((x - q)**2 * (x**2 - 1)**(1/2) * x ) \
            / ((1 + np.exp(-x * Z)) * (1 + np.exp( (x - q) * Z_nu )))

        x = np.linspace(1, 100, 1000)
        dx = x[1] - x[0]
        return 1/tau * np.trapz(f(x), dx=dx)

    def gamma_p_n(self, T, q=2.53):
        return self.gamma_n_p(T, -q)

    def pn_to_Dgamma(self, T):       # 1)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        p_n = 2.5e4 * rho_b
        lambda_D = 4.68e9 * p_n / rho_b * T9**(3 / 2) * np.exp(-25.82 / T9)

        return p_n, lambda_D

    def pD_to_He3gamma(self, T):     # 2)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        p_D = 2.23 * 10**3 * rho_b * T9**(-2/3) * np.exp(-3.72 * T9**(-1/3)) \
              * (1 + 0.112 * T9**(1/3) + 3.38 * T9**(2/3) + 2.65 * T9)
        lambda_He3 = 1.63 * 10**(10) * p_D / rho_b * T9**(3/2) * np.exp(-63.75 / T9)

        return p_D, lambda_He3

    def nD_to_Tgamma(self, T):       # 3)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        n_D = rho_b * (75.5 + 1250 * T9)
        lambda_T = 1.63 * 10**(10) * n_D /rho_b * T9**(3/2) * np.exp(-72.62 / T9)

        return n_D, lambda_T

    def nHe3_to_pT(self, T):         # 4)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        n_He3    = 7.06 * 10**8 * rho_b
        lambda_pT = n_He3 * np.exp(-8.864 / T9)

        return n_He3, lambda_pT

    def pT_to_He4gamma(self, T):     # 5)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        p_T = 2.87 * 10**4 * rho_b * T9**(-2/3) * np.exp(-3.87 * T9**(-1/3)) \
             * (1 + 0.108 * T9**(1/3) + 0.466 * T9**(2/3) + 0.352 * T9 \
             + 0.300 * T9**(4/3) + 0.576 * T9**(5/3))
        lambda_He4 = 2.59 * 10**(10) * p_T / rho_b * T9**(3/2) * np.exp(-229.9 / T9)

        return p_T, lambda_He4

    def nHe3_to_He4gamma(self, T):   # 6)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        n_He3 = 6.0 * 10**3 * rho_b * T9
        lambda_He4 = 2.60 * 10**(10) * n_He3 / rho_b * T9**(3/2) * np.exp(-238.8 / T9)

        return n_He3, lambda_He4

    def DD_to_nHe3(self, T):         # 7)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        D_D = 3.9 * 10**8 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) \
              * (1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.440 * T9)
        lambda_nHe3 = 1.73 * D_D * np.exp(-37.94 / T9)

        return D_D, lambda_nHe3

    def DD_to_pT(self, T):           # 8)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        D_D = 3.9 * 10**8 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) \
              * (1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.440 * T9)
        lambda_pT = 1.73 * D_D * np.exp(-46.80 / T9)

        return D_D, lambda_pT

    def DD_to_He4gamma(self, T):     # 9)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        D_D  = 24.1 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) \
               * (T9**(2/3) + 0.685 * T9 + 0.152 * T9**(4/3) + 0.265 * T9**(5/3))
        lambda_He4 = 4.50 * 10**(10) * D_D / rho_b * T9**(3/2) * np.exp(-276.7 / T9)

        return D_D, lambda_He4

    def DHe3_to_He4p(self, T):       # 10)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        D_He3 = 2.60 * 10**9 * rho_b * T9**(-3/2) * np.exp(-2.99 / T9)
        lambda_He4p = 5.50 * D_He3 * np.exp(-213.0 / T9)

        return D_He3, lambda_He4p

    def DT_to_He4n(self, T):         # 11)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        D_T = 1.38 * 10**9 * rho_b * T9**(-3/2) * np.exp(-0.745 / T9)
        lambda_He4n = 5.50 * D_T * np.exp(-204.1 / T9)

        return D_T, lambda_He4n

    def He3T_to_He4D(self, T):       # 15)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        He3_T = 3.88 * 10**9 * rho_b * T9**(-2/3) * np.exp(-7.72 * T9**(-1/3))\
                * (1 + 0.0540 * T9**(1/3))
        lambda_He4D = 1.59 * He3_T * np.exp(-166.2 / T9)

        return He3_T, lambda_He4D

    def He3He4_to_Be7gamma(self, T): # 16)

        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        He3_He4 = 4.8 * 10**6 * rho_b * T9**(-2/3) * np.exp(-12.8 * T9**(-1/3)) \
                  * (1 + 0.0326 * T9**(1/3) - 0.219 * T9**(2/3) - 0.0499 * T9 \
                  + 0.0258 * T9**(4/3) + 0.0150 * T9**(5/3))
        lambda_Be7 = 1.12 * 10**(10) *He3_He4 / rho_b * T9**(3/2) * np.exp(-18.42 / T9)

        return He3_He4, lambda_Be7

    def THe4_to_Li7gamma(self, T):   # 17)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        T_He4 = 5.28 * 10**5 * rho_b * T9**(-2/3) * np.exp(-8.08 * T9**(-1/3))\
                * (1 + 0.0516 * T9**(1/3))
        lambda_Li7 = 1.12 * 10**(10) *T_He4 / rho_b * T9**(3/2) * np.exp(-28.63 / T9)

        return T_He4, lambda_Li7

    def nBe7_to_pLi7(self, T):       # 18)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        n_Be7 = 6.74 * 10**9 * rho_b
        lambda_pLi7 = n_Be7 * np.exp(-19.07 / T9)

        return n_Be7, lambda_pLi7

    def pLi7_to_He4He4(self, T):     # 20)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        p_Li7 = 1.42 * 10**9 * rho_b * T9**(-2/3) * np.exp(-8.47 * T9**(-1/3))\
                * (1 + 0.0493 * T9**(1/3))
        lambda_2He4 = 4.64 * p_Li7 * np.exp(-201.3 / T9)

        return p_Li7, lambda_2He4

    def nBe7_to_He4He4(self, T):     # 21)
        T9 = T / 1e9; a = self.T0 / T
        rho_c0 = self.rho_c0; omega_b0 = self.omega_b0
        rho_b = omega_b0 * rho_c0 / a**3

        n_Be7 = 1.2 * 10**7 * rho_b * T9
        lambda_2He4 = 4.64 * n_Be7 * np.exp(-220.4 / T9)

        return n_Be7, lambda_2He4


ElementAbundance().task_j(6)
