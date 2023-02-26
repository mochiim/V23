import numpy as np
import matplotlib.pyplot as pyplot

class energy:

    """
    This class accepts a temperature [K] and density [kg/m^3] of the Sun

    The class contain the following functions:
    reaction_rates(self)- calculation of nuclear reaction rates in reactions per second and per [mole/m^3]
    energy_production(self)- calculates the amount of energy produced by the fusion chains (PP chain and CNO).
    _sanitycheck(self, exp)- sanity check to make sure the reaction rates are correct.


    """

    def __init__(self, T, rho):

        # defining useful variables (fetched from appendix A in lecture notes)
        self.N_A = 6.0221e23    # Avogadro's number [1/mol]
        self.mu = 1.6606e-27    # atomic mass unit [kg]
        self.eV = 1.6022e-19    # [J]
        self.c = 2.9979e8       # speed of light [ms^-1]

        # solar attributes (fetched from appendix B in lecture notes)
        self.T = T              # temperature of solar core [K]
        self.rho = rho          # density of solar core [kg/m^3]

        # mass fraction given in task description
        self.X = 0.7            # hydrogen
        self.Y = 0.29           # helium 4
        self.Y_3 = 1e-10        # helium 3
        self.Z_Li = 1e-7        # lithium 7
        self.Z_Be = 1e-7        # beryllium 7
        self.Z_14 = 1e-11       # nitrogen 14

        # energy production (fetched from table 3.2 and 3.3 in lecture notes)
        self.Q = np.zeros(7)                    # energy released per reaction [J]
        self.Q[0] = 1.177 + 5.494               # H + H (PP0) [MeV]
        self.Q[1] = 12.860                      # He3 + He3 (PP1) [MeV]
        self.Q[2] = 1.586                       # He3 + He4 (PP2, PP3) [MeV]
        self.Q[3] = 0.049                       # Be7 + e (PP2)
        self.Q[4]= 17.346                       # Li7 + H (PP2)
        self.Q[5] = 0.137 + 8.367 + 2.995       # Be7 + H (PP3)
        self.Q[6] = (1.944 + 1.513 + 7.551 + \
                7.297 + 1.757 + 4.966)          # CNO cycle [MeV]
        self.Q = self.Q*(self.eV*1e6)           # [MeV] -> [J]

        # number densities
        self.n_p = rho*self.X / self.mu
        self.n_He4 = rho*self.Y / (4*self.mu)
        self.n_He3 = rho*self.Y_3 / (3*self.mu)
        self.n_Be = rho*self.Z_Be / (7*self.mu)
        self.n_Li = rho*self.Z_Li / (7*self.mu)
        self.n_14 = rho*self.Z_14 / (14*self.mu)
        self.n_e = (self.n_p) + (self.n_He4 * 2) + (self.n_He3 * 2) + \
                   (self.n_Li*3) + (self.n_Be * 4) + (self.n_14*7)

        # reaction rates
        self.r_ = np.zeros(7)

        # energy production
        self.eps = np.zeros(7)       # [J/kgs]

        # proportionality function lambda
        self.lambdas = None         # [m^3/s]

        #  atomic mass of different elements
        self.m_p = 1.0073           # [u]
        self.m_H = 1.0078           # [u]
        self.m_He3 = 3.0160         # [u]
        self.m_He4 = 4.0026         # [u]
        self.m_e = 5.4858e-4        # [u]


    def reaction_rates(self):
        """
        Calculating reaction rates per unit mass for PP chain and CNO cycle
        """
        # defining variables
        r_ = self.r_
        T = self.T
        T9 = T*1e-9                     # solar core temperature in units 1e9 K
        T9x = T9 / (1 + 4.95e-2*T9)     # scaled temperature 1
        T9xx = T9 / (1 + .759*T9)       # scaled temperature 2
        N_A = self.N_A                  # Avodagros number

        # nuclear reaction rates in reactions per second and per [mole/m^3]
        # fetched from table 3.1 in lecture notes

        lambda_pp = (4.01e-15 * T9**(-2/3) * np.exp(-3.38 * T9**(-1/3)) * \
                    (1 + .123 * T9**(1/3) + 1.09 * T9**(2/3) + .938 * T9)) / (N_A*1e6)

        lambda_33 = (6.04e10 * T9**(-2/3) * np.exp(-12.276 * T9**(-1/3)) * \
                    (1 + .034 * T9**(1/3) - .522 * T9**(2/3) - .124 * T9 + .353\
                     * T9**(4/3) + .213 * T9**(5/3))) / (N_A*1e6)

        lambda_34 = (5.61e6 * T9x**(5/6) * T9**(-3/2) * np.exp(-12.826 * \
                    T9x**(-1/3))) / (N_A*1e6)

        lambda_e7 = (1.34e-10 * T9**(-1/2) * (1 - .537 * T9**(1/3) + 3.86 * \
                    T9**(2/3) + .0027 * T9**(-1) * np.exp(2.515e-3 * T9**(-1)))) / (N_A*1e6)

        lambda_17prime = (1.096e9 * T9**(-2/3) * np.exp(-8.472 * T9**(-1/3))\
                        - 4.83e8 * T9x**(5/6) * T9**(-3/2) * np.exp(-8.472 \
                        * T9xx**(-1/3)) + 1.06e10 * T9**(-3/2) * np.exp(-30.442 * T9**(-1))) / (N_A*1e6)

        lambda_17 = (3.11e5 * T9**(-2/3) * np.exp(-10.262 * T9**(-1/3)) + 2.53e3 \
                    * T9**(-2/3) * np.exp(-7.306 * T9**(-1))) / (N_A*1e6)

        lambda_p14 = (4.9e7 * T9**(-2/3) * np.exp(-15.228 * T9**(-1/3) - .092 \
                    * T9**2) * (1 + .027 * T9**(1/3) - .778 * T9**(2/3) - .149 \
                    * T9 + .261 * T9**(4/3) + .127 * T9**(5/3)) + 2.37e3 * T9**(-3/2)\
                     * np.exp(-3.011 * T9**(-1)) + 2.19e4 * np.exp(-12.53 * T9**(-1))) / (N_A*1e6)

        # including upper limit of Beryllium 7
        if self.T < 1e6:
            lambda_e7 = 1.57e-7/(self.n_e*N_A)

        # updating values of self.lambdas
        self.lambdas = np.array([lambda_pp, lambda_33, lambda_34, lambda_e7, lambda_17prime, lambda_17, lambda_p14])

        # calculating reaction rates [1/kgs]
        r_[0] = (self.n_p*self.n_p)*lambda_pp / (2*self.rho)         # H + H (PP0)
        r_[1] = (self.n_He3*self.n_He3)*lambda_33 / (2*self.rho)     # He3 + He3 (PP1)
        r_[2] = (self.n_He3*self.n_He4)*lambda_34 / (self.rho)       # He3 + He4 (PP2, P3)
        r_[3] = (self.n_Be*self.n_e)*lambda_e7 / (self.rho)          # Be7 + e (PP2)
        r_[4] = (self.n_Li*self.n_p)*lambda_17prime / (self.rho)     # Li7 + H (PP2)
        r_[5] = (self.n_Be*self.n_p)*lambda_17 / (self.rho)          # Be7 + H (PP3)
        r_[6] = (self.n_14*self.n_p)*lambda_p14 / (self.rho)         # CNO

        # making sure no step consumes more of an element than the previous step are able to produce

        # helium 3
        if r_[0] < (2*r_[1] + r_[2]):
            H = r_[0]/ (2*r_[1] + r_[2]) # normalizing factor
            r_[1] = H*r_[1]
            r_[2] = H*r_[2]

        # Beryllium 7
        if r_[2] < r_[3] + r_[5]:
            B = r_[2] / (r_[3] + r_[5]) # normalizing factor
            r_[3] = B*r_[3]
            r_[5] = B*r_[5]

        # Lithium 7
        if r_[3] < r_[4]:
            L = r_[3] / r_[4] # normalizing factor
            r_[4] = L*r_[4]

        # updating self.r_
        self.r_ = r_
        return None

    def energy_production(self):
        """
        Calculating energy generation per unit mass for all reactions
        """
        eps = self.eps
        for i in range(7):
            eps[i] = self.r_[i]*self.Q[i]

        print("Energy production energy output from each individual reaction in the PP chain and CNO cycle. ")
        print(f"PP0: {eps[0]:15.2} J/kgs")
        print(f"PP1: {eps[1]:15.2} J/kgs")
        print(f"PP2: {eps[2] + eps[3] + eps[4]:15.2} J/kgs")
        print(f"PP3: {eps[2] + eps[5]:15.2} J/kgs")
        print(f"CNO: {eps[6]:15.2} J/kgs")
        return None

    def energy_output(self):
        # expected energy release per He4 in PP chain (PPI, PP2, PP3) and CNO [MeV]
        exp = [26.202, 25.652, 18.598, 25.028]

        conversion = (self.mu*self.c**2*1e-6)/self.eV
        PPI = ((2*self.m_He3) - (self.m_He4 + 2*(self.m_p + self.m_e)))*conversion

        print(PPI)
        return None

    def _sanitycheck(self, exp):
        """
        Sanity check to see if results from calculation corresponds with the expected values.
        exp: a list containing the expectation Values
        """
        # defining variables
        r_ = self.r_
        Q = self.Q
        tol = 1.1
        rho = self.rho

        # expectation values [Jm^-3s^-1]
        exp0 = exp[0]
        exp1 = exp[1]
        exp2 = exp[2]
        exp3 = exp[3]
        exp4 = exp[4]
        exp5 = exp[5]
        exp6 = exp[6]

        #
        res0 = rho*r_[0]*Q[0]
        res1 = rho*r_[1]*Q[1]
        res2 = rho*r_[2]*Q[2]
        res3 = rho*r_[3]*Q[3]
        res4 = rho*r_[4]*Q[4]
        res5 = rho*r_[5]*Q[5]
        res6 = rho*r_[6]*Q[6]

        print(f"Sanity check for T = {self.T} K and rho = {self.rho} kg/m^3")
        print(" |  Results        |Expected Values          |Sanity test passed?")
        print(f" | {res0:15.2} | {exp0:15.2}         |   {res0/exp0 < tol}")
        print(f" | {res1:15.2} | {exp1:15.2}         |   {res1/exp1 < tol}")
        print(f" | {res2:15.3} | {exp2:15.2}         |   {res2/exp2 < tol}")
        print(f" | {res3:15.4} | {exp3:15.2}         |   {res3/exp3 < tol}")
        print(f" | {res4:15.5} | {exp4:15.2}         |   {res4/exp4 < tol}")
        print(f" | {res5:15.6} | {exp5:15.2}         |   {res5/exp5 < tol}")
        print(f" | {res6:15.7} | {exp6:15.2}         |   {res6/exp6 < tol}")

        return None

if __name__ == "__main__":

    T = 1.57*1e7    # temperature of solar core [K]
    rho = 1.62*1e5  # density of solar core [kg/m^3]
    T8 = 1e8        # [K]

    A = energy(T, rho)
    A.reaction_rates()
    #A._sanitycheck([4.04*1e2, 8.68*1e-9, 4.86*1e-5, 1.49*1e-6, 5.29*1e-4, 1.63*1e-6, 9.18*1e-8])
    #A.energy_production()
    A.energy_output()

    #B = energy(1e8, rho)
    #B.reaction_rates()
    #B._sanitycheck([7.34e4, 1.09e0, 1.74e4, 1.22e-3, 4.35e-1, 1.26e5, 3.45e4])
