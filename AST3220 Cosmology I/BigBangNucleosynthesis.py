import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.integrate import solve_ivp, quad

plt.style.use("seaborn")
plt.rcParams['font.size'] = 20
plt.rcParams["lines.linewidth"] = 2

class BigBangNucleosynthesis():
    def __init__(self):
        # defining useful constants in CGS units
        self.c = const.c.cgs.value               # speed of light in vacuum
        self.k = const.k_B.cgs.value             # Boltzmann constant
        self.hbar = const.hbar.cgs.value         # reduced Planck constant
        self.G = const.G.cgs.value               # Newton's gravitational constant
        self.m_e = const.m_e.cgs.value           # electron mass
        self.m_p = const.m_p.cgs.value           # proton mass
        self.m_n = const.m_n.cgs.value           # neutron mass

        self.h = 0.7                             # Hubble constant [-]
        self.H0 = 100*self.h*1e5/3.09e+24        # Hubble parameter in CGS units
        self.T0 = 2.72548                        # temperature today of CMB [K]
        self.N_eff = 3                           # neutrino species [-]
        self.Omega_b0 = 0.05
        self.rho_c0 = 1.879e-29*self.h**2        # critial density [gcm^-3]

# Task f
    def differentials_f(self, logT, Y):
        """
        This function includes differential equations for dY_i/d(lnT) for i = n, p.
        The differential equations are given in eq. (10) and (11) in the task description.
        """
        T = np.exp(logT)
        a = self.T0/T                                   # scale factor

        Omega_r0 = (8*np.pi**3*self.G*(self.k*self.T0)**4)/(45*self.H0**2*self.hbar**3*self.c**5)*(1 + (7/8)*self.N_eff*(4/11)**(4/3))
        H = self.H0*np.sqrt(Omega_r0)*a**(-2)           # eq. 14 in task description
        Y_n = Y[0]; Y_p = Y[1]                          # unpack values from Y and assign names

        dY_n = -1 / H*(Y_p*self.lambda_w(T, "proton to neutron") - Y_n*self.lambda_w(T, "neutron to proton") )
        dY_p = -1 / H*(Y_n*self.lambda_w(T, "neutron to proton") - Y_p*self.lambda_w(T, "proton to neutron") )

        return np.array([dY_n, dY_p])

    def taskf(self):
        """
        Solving differential equations for dY_i/d(lnT) for i = n, p and plotting results.
        """
        # defining temperature interval
        T_i = 100e9; T_f = .1e9                         # [K]
        Y_n_Ti = (1 + np.exp((self.m_n - self.m_p)*self.c**2/self.k*T_i ))**(-1) # from task e)
        Y_p_Ti = 1 - Y_n_Ti                             # from task e)

        solve = solve_ivp(self.differentials_f, [np.log(T_i), np.log(T_f)], [Y_n_Ti, Y_p_Ti], method= "Radau", rtol = 1e-12, atol = 1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]; Y_p = solve.y[1]              # unpack values and assign names

        plt.plot(T, Y_p, label = "p", color = "blue")   # plotting solutions for Y_p
        plt.plot(T, Y_n, label = "n", color = "orange") # plotting solutions for Y_n

        # equilibrium values
        plt.plot(T, 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k*T))), linestyle = "dotted", color = "tab:orange")
        plt.plot(T, 1 - 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k*T))), linestyle = "dotted", color = "tab:blue")
        plt.xscale("log"); plt.yscale("log")

        plt.xlim([T[0], T[-1]]); plt.ylim([1e-3, 2])    # matching the plotting window of figure we are attempting to reproduce
        plt.xlabel('$T$ [K]'); plt.ylabel(r'$Y_i$')     # x and y label of plotting window
        plt.legend()
        #plt.savefig("f.png")

# Task h
    def differentials_h(self, logT, Y):
        """
        This function includes differential equations for dY_i/d(lnT) for i = n, p, D.
        The differential equations are given in eq. (20), (21), and (22) in the task description.
        """
        T = np.exp(logT)
        a = self.T0/T

        Omega_r0 = (8*np.pi**3*self.G*(self.k*self.T0)**4)/(45*self.H0**2*self.hbar**3*self.c**5)*(1 + (7/8)*self.N_eff*(4/11)**(4/3))
        H = self.H0*np.sqrt(Omega_r0)*a**(-2)
        Y_n = Y[0]; Y_p = Y[1]; Y_D = Y[2]

        pn, lambda_gamma_D = self.lambda_gamma_D(T)
        dY_n = -1/H * (- Y_n*self.lambda_w(T, "neutron to proton") + Y_p*self.lambda_w(T, "proton to neutron") + lambda_gamma_D*Y_D - pn*Y_n*Y_p)
        dY_p = -1/H * (- Y_p*self.lambda_w(T, "proton to neutron") + Y_n*self.lambda_w(T, "neutron to proton") + lambda_gamma_D*Y_D - pn*Y_n*Y_p)
        dY_D = -1/H*(- lambda_gamma_D*Y_D + pn*Y_n*Y_p)

        return np.array([dY_n, dY_p, dY_D])

    def taskh(self):
        T_i = 100e9; T_f = .1e9
        Y_n_Ti = (1 + np.exp((self.m_n - self.m_p)*self.c**2/self.k*T_i ))**(-1) # from task e)
        Y_p_Ti = 1 - Y_n_Ti
        Y_D_Ti = 0

        solve = solve_ivp(self.differentials_h, [np.log(T_i), np.log(T_f)], [Y_n_Ti, Y_p_Ti, Y_D_Ti], method = "Radau", rtol = 1e-12, atol = 1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]; Y_p = solve.y[1]; Y_D = solve.y[2]

        plt.plot(T, Y_p, label = "p", color = "blue")
        plt.plot(T, Y_n, label = "n", color = "orange")
        plt.plot(T, 2*Y_D, label = 'D', color = "green")
        plt.plot(T, 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k*T))), linestyle = "dotted", color = "tab:orange")
        plt.plot(T, 1 - 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k*T))), linestyle = "dotted", color = "tab:blue")
        plt.xscale("log"); plt.yscale("log")
        plt.xlim([T[0], T[-1]]); plt.ylim([1e-3, 2])
        plt.xlabel('$T$ [K]'); plt.ylabel(r'Mass fraction $A_iY_i$')
        plt.legend()
        #plt.savefig("h.png")

# Task i
    def differentials_i(self, logT, Y):
        T = np.exp(logT)
        a = self.T0 / T

        Omega_r0 = (8*np.pi**3*self.G*(self.k*self.T0)**4)/(45*self.H0**2*self.hbar**3*self.c**5)*(1 + (7/8)*self.N_eff*(4/11)**(4/3))
        H = self.H0*np.sqrt(Omega_r0)*a**(-2)

        Y_n = Y[0]; Y_p = Y[1]; Y_D = Y[2]; Y_T = Y[3]; Y_He3 = Y[4]; Y_He4 = Y[5]; Y_Li7 = Y[6]; Y_Be7 = Y[7]

        dY_n, dY_p, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7 = 0, 0, 0, 0, 0, 0, 0, 0

        # a), 1)- 3)
        dY_n = self.differentials_f(logT, Y)[0]
        dY_p = self.differentials_f(logT, Y)[1]

        # b) 1)
        dY_n = self.differentials_h(logT, Y)[0]
        dY_p = self.differentials_h(logT, Y)[1]
        dY_D = self.differentials_h(logT, Y)[2]

        # b) 2)
        pD, lambda_gamma_He3 = self.reactionb2(T)
        dY_p = dY_p - 1/H*(- Y_p*Y_D*pD + Y_He3*lambda_gamma_He3)
        dY_D = dY_D - 1/H*(- Y_p*Y_D*pD + Y_He3*lambda_gamma_He3)
        dY_He3 = dY_He3 - 1/H*( Y_p*Y_D*pD - Y_He3*lambda_gamma_He3)

        # b) 3)
        nD, lambda_gamma_T = self.reactionb3(T)
        dY_n = dY_n - 1/H * (Y_D * Y_T * lambda_gamma_T - Y_n * Y_D * nD)
        dY_D = dY_D - 1/H * (Y_D * Y_T * lambda_gamma_T - Y_n * Y_D * nD)
        dY_T = dY_T - 1/H * (Y_n * Y_D * nD - Y_T * lambda_gamma_T)

        # b) 4)
        nHe3, pT = self.reactionb4(T)
        dY_n = dY_n     - 1/H * (Y_p * Y_T * pT - Y_n * Y_He3 * nHe3)
        dY_He3 = dY_He3 - 1/H * (Y_p * Y_T * pT - Y_n * Y_He3 * nHe3)
        dY_p = dY_p     - 1/H * (Y_n * Y_He3 * nHe3 - Y_p * Y_T * pT)
        dY_T = dY_T     - 1/H * (Y_n * Y_He3 * nHe3 - Y_p * Y_T * pT)

        # b) 5)
        pT, lambda_gamma_He4 = self.reactionb5(T)
        dY_p = dY_p     - 1/H * (Y_He4 * lambda_gamma_He4 - Y_p * Y_T * pT)
        dY_T = dY_T     - 1/H * (Y_He4 * lambda_gamma_He4 - Y_p * Y_T * pT)
        dY_He4 = dY_He4 - 1/H * (Y_p * Y_T * pT - Y_He4 * lambda_gamma_He4)

        # b) 6)
        nHe3, lambda_gamma_He4 = self.reactionb6(T)
        dY_n = dY_n     - 1/H * (Y_He4 * lambda_gamma_He4 - Y_n * Y_He3 * nHe3)
        dY_He3 = dY_He3 - 1/H * (Y_He4 * lambda_gamma_He4 - Y_n * Y_He3 * nHe3)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * lambda_gamma_He4 - Y_n * Y_He3 * nHe3)

        # b) 7)
        DD, nHe3 = self.reactionb7(T)
        dY_D = dY_D     - 1/H * (Y_n * Y_He3 * nHe3 - Y_D * Y_D * DD)
        dY_n = dY_n     + 1/H * (Y_n * Y_He3 * nHe3 - 0.5 * Y_D**2 * DD)
        dY_He3 = dY_He3 + 1/H * (Y_n * Y_He3 * nHe3 - 0.5 * Y_D**2 * DD)

        # b) 8)
        DD, pT = self.reactionb8(T)
        dY_D = dY_D - 1/H * (Y_p * Y_T * pT - Y_D**2 * DD)
        dY_p = dY_p + 1/H * (Y_p * Y_T * pT - 0.5 * Y_D**2 * DD)
        dY_T = dY_T + 1/H * (Y_p * Y_T * pT - 0.5 * Y_D**2 * DD)

        # b) 9)
        DD, lambda_gamma_He4 = self.reactionb9(T)
        dY_D = dY_D     - 1/H * (Y_He4 * lambda_gamma_He4 - Y_D**2 * DD)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * lambda_gamma_He4 - 0.5 * Y_D**2 * DD)

        # b) 10)
        DHe3, He4p = self.reactionb10(T)
        dY_D = dY_D     - 1/H * (Y_He4 * Y_p * He4p - Y_D * Y_He3 * DHe3)
        dY_He3 = dY_He3 - 1/H * (Y_He4 * Y_p * He4p - Y_D * Y_He3 * DHe3)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * Y_p * He4p - Y_D * Y_He3 * DHe3)
        dY_p = dY_p     + 1/H * (Y_He4 * Y_p * He4p - Y_D * Y_He3 * DHe3)

        # b) 11)
        DT, He4n = self.reactionb11(T)
        dY_D = dY_D     - 1/H * (Y_He4 * Y_n * He4n - Y_D * Y_T * DT)
        dY_T = dY_T     - 1/H * (Y_He4 * Y_n * He4n - Y_D * Y_T * DT)
        dY_He4 = dY_He4 - 1/H * -(Y_He4 * Y_n * He4n - Y_D * Y_T * DT)
        dY_n = dY_n     - 1/H * -(Y_He4 * Y_n * He4n - Y_D * Y_T * DT)

        # b) 15)
        He3T, He4D = self.reactionb15(T)
        dY_He3 = dY_He3 - 1/H * (Y_He4 * Y_D * He4D - Y_He3 * Y_T * He3T)
        dY_T = dY_T     - 1/H * (Y_He4 * Y_D * He4D - Y_He3 * Y_T * He3T)
        dY_D = dY_D     + 1/H * (Y_He4 * Y_D * He4D - Y_He3 * Y_T * He3T)
        dY_He4 = dY_He4 + 1/H * (Y_He4 * Y_D * He4D - Y_He3 * Y_T * He3T)

        # b) 16)
        He3He4, lambda_gamma_Be7 = self.reactionb16(T)
        dY_He3 = dY_He3 - 1/H * (Y_Be7 * lambda_gamma_Be7 - Y_He3 * Y_He4 * He3He4)
        dY_He4 = dY_He4 - 1/H * (Y_Be7 * lambda_gamma_Be7 - Y_He3 * Y_He4 * He3He4)
        dY_Be7 = dY_Be7 + 1/H * (Y_Be7 * lambda_gamma_Be7 - Y_He3 * Y_He4 * He3He4)

        # b) 17)
        THe4, lambda_gamma_Li7 = self.reactionb17(T)
        dY_T = dY_T     - 1/H * (Y_Li7 * lambda_gamma_Li7 - Y_T * Y_He4 * THe4)
        dY_He4 = dY_He4 - 1/H * (Y_Li7 * lambda_gamma_Li7 - Y_T * Y_He4 * THe4)
        dY_Li7 = dY_Li7 + 1/H * (Y_Li7 * lambda_gamma_Li7 - Y_T * Y_He4 * THe4)

        # b) 18)
        nBe7, pLi7 = self.reactionb18(T)
        dY_n = dY_n     - 1/H * (Y_p * Y_Li7 * pLi7 - Y_n * Y_Be7 * nBe7)
        dY_Be7 = dY_Be7 - 1/H * (Y_p * Y_Li7 * pLi7 - Y_n * Y_Be7 * nBe7)
        dY_p = dY_p     + 1/H * (Y_p * Y_Li7 * pLi7 - Y_n * Y_Be7 * nBe7)
        dY_Li7 = dY_Li7 + 1/H * (Y_p * Y_Li7 * pLi7 - Y_n * Y_Be7 * nBe7)

        # b) 20)
        pLi7, He4He4 = self.reactionb20(T)
        dY_p = dY_p     - 1/H * (0.5 * Y_He4**2 * He4He4 - Y_p * Y_Li7 * pLi7)
        dY_Li7 = dY_Li7 - 1/H * (0.5 * Y_He4**2 * He4He4 - Y_p * Y_Li7 * pLi7)
        dY_He4 = dY_He4 + 1/H * (Y_He4**2 * He4He4 - Y_p * Y_Li7 * pLi7)

        # b) 21)
        nBe7, He4He4 = self.reactionb21(T)
        dY_n = dY_n     - 1/H * (0.5 * Y_He4**2 * He4He4 - Y_n * Y_Be7 * nBe7)
        dY_Be7 = dY_Be7 - 1/H * (0.5 * Y_He4**2 * He4He4 - Y_n * Y_Be7 * nBe7)
        dY_He4 = dY_He4 + 1/H * (Y_He4**2 * He4He4 - Y_n * Y_Be7 * nBe7)


        return np.array([dY_n, dY_p, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7])

    def taski(self):
        T_i = 100e9; T_f = .01e9
        Y_n_Ti = (1 + np.exp((self.m_n - self.m_p)*self.c**2/self.k*T_i ))**(-1) # from task e)
        Y_p_Ti = 1 - Y_n_Ti

        YD, YT, YHe3, YHe4, YLi7, YBe7 = 0, 0, 0, 0, 0, 0

        solve = solve_ivp(self.differentials_i, t_span=[np.log(T_i), np.log(T_f)], y0=[Y_n_Ti, Y_p_Ti, YD, YT, YHe3, YHe4, YLi7, YBe7], method='Radau', rtol=1e-12, atol=1e-12)

        T = np.exp(solve.t)
        Y_n = solve.y[0]
        Y_p = solve.y[1]
        Y_D = solve.y[2]
        Y_T = solve.y[3]
        Y_He3 = solve.y[4]
        Y_He4 = solve.y[5]
        Y_Li7 = solve.y[6]
        Y_Be7 = solve.y[7]

        plt.plot(T, Y_p, label = "p")
        plt.plot(T, Y_n, label = "n")
        plt.plot(T, 2*Y_D, label = "D")
        plt.plot(T, 3*Y_T, label = "T")
        plt.plot(T, 3*Y_He3, label = r"He$^3$")
        plt.plot(T, 4*Y_He4, label = r"He$^4$")
        plt.plot(T, 7*Y_Li7, label = r"Li$^7$")
        plt.plot(T, 7*Y_Be7, label = r"Be$^7$")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("$T$ [K]"); plt.ylabel(r"Mass fraction $A_i Y_i$")
        plt.xlim([T[0], T[-1]]); plt.ylim([1e-11, 1e1])
        plt.legend()
        plt.tight_layout()
        # plt.savefig('i.png')

########## reaction rates from table 2 in "ON THE SYNTHESIS OF ELEMENTS AT VERY HIGH TEMPERATURES" ##########
    def lambda_w(self, T, type):
        """
        Table 2, a) weak interactions, reactions 1) - 3)
        """
        T_nu = (4/11)**(1/3)*T # from task b)
        Z = self.m_e*self.c**2/(self.k*T)
        Z_nu = self.m_e*self.c**2/(self.k*T_nu)
        tau = 1700 # free neutron decay time [s]

        if type == "neutron to proton":
            q = 2.53
        elif type == "proton to neutron":
            q = - 2.53

        f = lambda x: ((x+q)**2*(x**2-1)**(1/2)*x)/((1 + np.exp(x*Z))*(1 + np.exp(-(x+q)*Z_nu ))) \
            + ((x-q)**2*(x**2-1)**(1/2)*x)/((1 + np.exp(-x*Z))*(1 + np.exp((x - q)*Z_nu)))

        x = np.linspace(1, 100, 1000)
        return 1/tau*np.trapz(f(x), dx = (x[1] - x[0]))

    def lambda_gamma_D(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 1)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        pn = 2.5e4*rho_b
        lambda_gamma = 4.68e9*pn/rho_b*T9**(3/2)*np.exp(- 25.82/T9)

        return pn, lambda_gamma

    def reactionb2(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 2)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        pD = 2.23 * 10**3 * rho_b * T9**(-2/3) * np.exp(-3.72 * T9**(-1/3)) \
              * (1 + 0.112 * T9**(1/3) + 3.38 * T9**(2/3) + 2.65 * T9)
        lambda_gamma_He3 = 1.63 * 10**(10) * pD / rho_b * T9**(3/2) * np.exp(-63.75 / T9)

        return pD, lambda_gamma_He3

    def reactionb3(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 3)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        nD = rho_b * (75.5 + 1250 * T9)
        lambda_gamma_T = 1.63 * 10**(10) * nD /rho_b * T9**(3/2) * np.exp(-72.62 / T9)

        return nD, lambda_gamma_T

    def reactionb4(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 4)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        nHe3    = 7.06 * 10**8 * rho_b
        pT = nHe3 * np.exp(-8.864 / T9)

        return nHe3, pT

    def reactionb5(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 5)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        pT = 2.87 * 10**4 * rho_b * T9**(-2/3) * np.exp(-3.87 * T9**(-1/3)) \
             * (1 + 0.108 * T9**(1/3) + 0.466 * T9**(2/3) + 0.352 * T9 \
             + 0.300 * T9**(4/3) + 0.576 * T9**(5/3))
        lambda_gamma_He4 = 2.59 * 10**(10) * pT / rho_b * T9**(3/2) * np.exp(-229.9 / T9)

        return pT, lambda_gamma_He4

    def reactionb6(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 6)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        nHe3 = 6.0 * 10**3 * rho_b * T9
        lambda_gamma_He4 = 2.60 * 10**(10) * nHe3 / rho_b * T9**(3/2) * np.exp(-238.8 / T9)

        return nHe3, lambda_gamma_He4

    def reactionb7(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 7)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        DD = 3.9 * 10**8 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) \
              * (1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.440 * T9)
        nHe3 = 1.73 * DD * np.exp(-37.94 / T9)

        return DD, nHe3

    def reactionb8(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 8)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        DD = 3.9 * 10**8 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) \
              * (1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.440 * T9)
        pT = 1.73 * DD * np.exp(-46.80 / T9)

        return DD, pT

    def reactionb9(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 9)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        DD  = 24.1 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) \
               * (T9**(2/3) + 0.685 * T9 + 0.152 * T9**(4/3) + 0.265 * T9**(5/3))
        lambda_gamma_He4 = 4.50 * 10**(10) * DD / rho_b * T9**(3/2) * np.exp(-276.7 / T9)

        return DD, lambda_gamma_He4

    def reactionb10(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 10)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        DHe3 = 2.60 * 10**9 * rho_b * T9**(-3/2) * np.exp(-2.99 / T9)
        He4p = 5.50 * DHe3 * np.exp(-213.0 / T9)

        return DHe3, He4p

    def reactionb11(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 11)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        DT = 1.38 * 10**9 * rho_b * T9**(-3/2) * np.exp(-0.745 / T9)
        He4n = 5.50 * DT * np.exp(-204.1 / T9)

        return DT, He4n

    def reactionb15(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 15)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        He3T = 3.88 * 10**9 * rho_b * T9**(-2/3) * np.exp(-7.72 * T9**(-1/3))\
                * (1 + 0.0540 * T9**(1/3))
        He4D = 1.59 * He3T * np.exp(-166.2 / T9)

        return He3T, He4D

    def reactionb16(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 16)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        He3He4 = 4.8 * 10**6 * rho_b * T9**(-2/3) * np.exp(-12.8 * T9**(-1/3)) \
                  * (1 + 0.0326 * T9**(1/3) - 0.219 * T9**(2/3) - 0.0499 * T9 \
                  + 0.0258 * T9**(4/3) + 0.0150 * T9**(5/3))
        lambda_gamma_Be7 = 1.12 * 10**(10) *He3He4 / rho_b * T9**(3/2) * np.exp(-18.42 / T9)

        return He3He4, lambda_gamma_Be7

    def reactionb17(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 17)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        THe4 = 5.28 * 10**5 * rho_b * T9**(-2/3) * np.exp(-8.08 * T9**(-1/3))\
                * (1 + 0.0516 * T9**(1/3))
        lambda_gamma_Li7 = 1.12 * 10**(10) *THe4 / rho_b * T9**(3/2) * np.exp(-28.63 / T9)

        return THe4, lambda_gamma_Li7

    def reactionb18(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 18)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        nBe7 = 6.74 * 10**9 * rho_b
        pLi7 = nBe7 * np.exp(-19.07 / T9)

        return nBe7, pLi7

    def reactionb20(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 20)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        pLi7 = 1.42 * 10**9 * rho_b * T9**(-2/3) * np.exp(-8.47 * T9**(-1/3))\
                * (1 + 0.0493 * T9**(1/3))
        He4He4 = 4.64 * pLi7 * np.exp(-201.3 / T9)

        return pLi7, He4He4

    def reactionb21(self, T):
        """
        Table 2, b) Strong and electromagnetic interactions, reaction 21)
        """
        a = self.T0/T
        T9 = T/1e9
        rho_b = self.Omega_b0*self.rho_c0/a**3

        nBe7 = 1.2 * 10**7 * rho_b * T9
        He4He4 = 4.64 * nBe7 * np.exp(-220.4 / T9)

        return nBe7, He4He4


#BigBangNucleosynthesis().taskf()
#BigBangNucleosynthesis().taskh()
BigBangNucleosynthesis().taski()
plt.show()
