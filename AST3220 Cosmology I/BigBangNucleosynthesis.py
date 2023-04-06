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
        self.hbar = const.hbar.cgs.value         # reducced Planck constant
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

""" Task f """
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

        dY_n = -1 / H*(Y_p*self.gamma(T, "proton to neutron") - Y_n*self.gamma(T, "neutron to proton") )
        dY_p = -1 / H*(Y_n*self.gamma(T, "neutron to proton") - Y_p*self.gamma(T, "proton to neutron") )

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

""" Task h """
    def differentials_h(self, logT, Y):
        T = np.exp(logT)
        a = self.T0/T
        T_9 = T/1e9

        Omega_r0 = (8*np.pi**3*self.G*(self.k*self.T0)**4)/(45*self.H0**2*self.hbar**3*self.c**5)*(1 + (7/8)*self.N_eff*(4/11)**(4/3))
        H = self.H0*np.sqrt(Omega_r0)*a**(-2)
        Y_n = Y[0]; Y_p = Y[1]; Y_D = Y[2]

        rho_b = self.Omega_b0*self.rho_c0/a**3
        pn = 2.5e4*rho_b
        lmbda = 4.68e9*pn/rho_b*T_9**(3/2)*np.exp(- 25.82/T_9)

        dY_n = -1/H * (- Y_n*self.gamma(T, "neutron to proton") + Y_p*self.gamma(T, "proton to neutron") + lmbda*Y_D - pn*Y_n*Y_p)
        dY_p = -1/H * (- Y_p*self.gamma(T, "proton to neutron") + Y_n*self.gamma(T, "neutron to proton") + lmbda*Y_D - pn*Y_n*Y_p)
        dY_D = -1/H*(- lmbda*Y_D + pn*Y_n*Y_p)

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

# reaction rates frm table 2 in "ON THE SYNTHESIS OF ELEMENTS AT VERY HIGH TEMPERATURES"
    def gamma(self, T, type):
        """
        Decay rates (eq. 12 in task description) from table 2 for weak interactions
        """
        T_nu = (4/11)**(1/3)*T # from task b)
        Z = 5.93/T
        Z_nu = 5.93/T_nu
        tau = 1700 # free neutron decay time [s]

        if type == "neutron to proton":
            q = 2.53
        elif type == "proton to neutron":
            q = - 2.53

        f = lambda x: ((x+q)**2*(x**2-1)**(1/2)*x)/((1 + np.exp(x*Z))*(1 + np.exp(-(x+q)*Z_nu ))) \
            + ((x-q)**2*(x**2-1)**(1/2)*x)/((1 + np.exp(-x*Z))*(1 + np.exp((x - q)*Z_nu)))

        x = np.linspace(1, 100, 1000)
        return 1/tau*np.trapz(f(x), dx = (x[1] - x[0]))


#BigBangNucleosynthesis().taskf()
BigBangNucleosynthesis().taskh()
plt.show()
