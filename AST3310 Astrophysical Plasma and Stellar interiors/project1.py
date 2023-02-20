import numpy as np
import matplotlib.pyplot as pyplot

class energy:
    """
    A class which accepts a temperature and density, and then calculates the
    amount of energy produced by the fusion chains discussed in Chapter 3
    """
    def __init__(self, T, rho, sanity = None):

        # defining some variables
        self.N_A = 6.0221*1e23   # Avogadro's number [1/mol]
        self.mu = 1.6606*1e-27  # atomic mass unit [kg]
        self.eV = 1.6022*1e-19  # [J]
        self.T = T              # temperature of solar core [K]
        self.rho = rho          # density of solar core [kg/m^3]


        # mass fraction
        self.X = 0.7            # hydrogen
        self.Y = 0.29           # helium 4
        self.Y_3 = 1e-10        # helium 3
        self.Z_Li = 1e-7        # lithium 7
        self.Z_Be = 1e-7        # beryllium 7
        self.Z_14 = 1e-11       # nitrogen 14

        # energy production
        self.Q = np.zeros(7)                 # energy released per reaction [J]
        self.Q[0] = 1.177 + 5.494            # PP0 [MeV]
        self.Q[1] = 12.860                   # PP1 [MeV]
        self.Q[2] = 1.586                    # PPII and PPIII [MeV]
        self.Q[3] = 0.049                    # PPII [MeV]
        self.Q[4]= 17.346                    # PPII [MeV]
        self.Q[5] = 0.137 + 8.367 + 2.995    # PPIII [MeV]
        self.Q[6] = (1.944 + 1.513 + 7.551 + \
                7.297 + 1.757 + 4.966)  # CNO cycle [MeV]
        self.Q = self.Q*(self.eV*1e6)          # [MeV] -> [J]

        # number densities
        self.n_p = rho*self.X / self.mu
        self.n_e = rho*(2*self.mu)*(1 + self.X)
        self.n_He4 = rho*self.Y / (4*self.mu)
        self.n_He3 = rho*self.Y_3 / (3*self.mu)
        self.n_Be = rho*self.Z_Be / (7*self.mu)
        self.n_Li = rho*self.Z_Li / (7*self.mu)
        self.n_14 = rho*self.Z_14 / (13*self.mu)

        # reaction rates
        self.r_ = np.zeros(7)

        # sanity test
        self.sanity = sanity

        def _energy_generation(self):
            """
            Energy generation per unit mass for all reactions
            """
            r_ = self.r_
            self.eps = np.zeros(7)       # [J/kgs]
            T = self.T*1e-9             # solar core temperature in units 1e9 K
            T1 = T/(1 + 4.95*1e-2*T)    # scaled temperature 1
            T2 = T/(1 + 0.759*T)        # scaled temperature 2
            N_A = self.N_A


            # nuclear reaction rates in reactions per second and per (mole/m^3)

            lambda_pp = 4.01*1e-15*T**(-2/3)*np.exp(-3.380*T**(-1/3)) \
                        * (1 + 0.123*T**(1/3) + 1.09*T**(2/3) + 0.938*T)/(N_A*1e6)

            lambda_33 = 6.04*1e10*T**(-2/3)*np.exp(-12.276*T**(-1/3)) \
                        * (1 + 0.034*T**(1/3)- 0.522*T**(2/3) - 0.124*T \
                         + 0.353*T**(4/3) + 0.213*T**(5/3)) / (N_A*1e6)

            lambda_34 = 5.61*1e6*T1**(5/6)*T**(-3/2)*np.exp(-12.826*T1**(-1/3))

            lambda_e7 = 1.34*1e-10*T**(-1/2)*(1 - 0.537*T**(1/3) + 3.86*T**(2/3)\
                        + 0.0027*T**(-1)*np.exp(2.515*1e-3*T*1e-1)) / (N_A*1e6)

            lambda_17_ = 1.096*1e9*T**(-2/3)*np.exp(-8.472*T**(-1/3))\
                        - 4.830*1e8*T2**(5/6)*T**(-3/2)*np.exp(-8472*T2**(-1/3))\
                        + 1.06*1e10*T**(-3/2)*np.exp(-30.422*T*1e-1) / (N_A*1e6)

            lambda_17 = 3.11*1e5*T**(-2/3)*np.exp(-10.262*T**(-1/3))\
                        + 2.53*1e3*T**(-3/2)*np.exp(-7.306*T*1e-1) / (N_A*1e6)

            lambda_p14 = 4.90*1e7*T**(-2/3)*np.exp(-15-228*T**(-1/3))\
                        + (1 + 0.027*T**(1/3) - 0.778*T**(2/3) - 0.149*T \
                        + 0.261*T**(4/3) + 0.127*T**(5/3)) \
                        + 2.37*1e3*T**(-3/2)*np.exp(-3.001*T*1e-1) \
                        + 2.19*1e4*np.exp(-12.53*T*1e-1) / (N_A*1e6)

            # Include upper limit of Beryllium 7
            if lambda_e7 > 1.57*1e-7/(self.n_e) and T < 1e6:
                lambda_e7 = 1.57*1e-7/(self.n_e*N_A)


            r_[0] = self.n_p**2*lambda_pp / (2*self.rho)
            r_[1] = self.n_He3**2*lambda_33 / (2*self.rho)
            r_[2] = self.n_He3*self.n_He4*lambda_34 / (self.rho)
            r_[3] = self.n_Be*self.n_e*lambda_e7 / (self.rho)
            r_[4] = self.n_Li*self.n_p*lambda_17_ / (self.rho)
            r_[5] = self.n_Be*self.n_p*lambda_17 / (self.rho)
            r_[6] = self.n_14*self.n_p*lambda_p14 / (self.rho)

            self.r_ = r_
            return self.r_

        def _sanitytest(self):
            r_ = self.r_
            Q = self.Q
            tol = 1
            rho = self.rho

            # expectation values [Jm^-3s^-1]
            exp0 = 4.04*1e2
            exp1 = 8.68*1e-9
            exp2 = 4.86*1e-5
            exp3 = 1.49*1e-6
            exp4 = 5.29*1e-4
            exp5 = 1.63*1e-6
            exp6 = 9.18*1e-8

            res0 = rho*r_[0]*Q[0]
            res1 = rho*r_[1]*Q[1]
            res2 = rho*r_[2]*Q[2]
            res3 = rho*r_[3]*Q[3]
            res4 = rho*r_[4]*Q[4]
            res5 = rho*r_[5]*Q[5]
            res6 = rho*r_[6]*Q[6]

            print(exp0, res0, abs(exp0 - res0) < tol)
            print(exp1, res1, abs(exp1 - res1) < tol)
            print(exp2, res2, abs(exp2 - res2) < tol)
            print(exp3, res3, abs(exp3 - res3) < tol)
            print(exp4, res4, abs(exp4 - res4) < tol)
            print(exp5, res5, abs(exp5 - res5) < tol)
            print(exp6, res6, abs(exp6 - res6) < tol)

            print("Prints values of sanity check")
            print("  | Results        |Expected Values          |True/False")
            print(f" | {res0:15.2} | {exp0:15.2}         |   {abs(res0 - exp0) < tol}")
            print(f" | {res1:15.2} | {exp1:15.2}         |   {abs(res1 - exp1) < tol}")
            print(f" | {res2:15.3} | {exp2:15.2}         |   {abs(res2 - exp2) < tol}")
            print(f" | {res2:15.4} | {exp3:15.2}         |   {abs(res3 - exp3) < tol}")
            print(f" | {res2:15.5} | {exp4:15.2}         |   {abs(res4 - exp4) < tol}")
            print(f" | {res2:15.6} | {exp5:15.2}         |   {abs(res5 - exp5) < tol}")
            print(f" | {res2:15.7} | {exp6:15.2}         |   {abs(res6 - exp6) < tol}")
            return None

        _energy_generation(self)
        #print(self.r_)
        self.sanity = _sanitytest(self)

        #print(_energy_generation(self))


A = energy(1.57*1e7, 1.62*1e5)



# Temperature of solar core T = 1.57*1e7 [K]
# Density of solar core rho = 1.62*1e5 [kg/m^3]
# Adjusted temperature T = 1e8 [K]
