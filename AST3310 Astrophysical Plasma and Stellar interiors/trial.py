""""" Skrevet for Python 3 """

"""
Nyttig info:
1 MeV   = 1.6021766 * 10^-13 J
"""

from numpy import pi, sqrt, exp, zeros, sum as Sum
from math import log10

class Energy():
    """
    Klasse for å regne ut energiptoduksjonen i et punkt i en stjerne.
    Tar inn variablene:
    T       = Temperaturen i punktet
    rho     = Massetettheten i punktet
    sanity  = En parameter som er enten "J" eller "Nei" og bestemmer om testing av klassen er slått på.
    Inneholder funksjonenen:
    - _Sanitytest(self)
    - _Energiproduksjon(self)
    - Kalk(self)
    Klassen kjøres ved å lage et klasseobjekt og kalle på funksjonen Kalk(). Da returnerer klassen
    eps[4]  - Den totale energiproduksjonen til punktet per masse, i ett tidspunkt.
    """

    def __init__(self, T, rho, sanity = None):
        """  Definerer variabler """

        m_u         = 1.660539040 * 10 ** (-27)     # Masse til proton [kg]
        R           = 6.957 * 10 ** 8               # Radius til sol[m]
        self.elem   = 1.602176634 * 10 ** (-19)     # Elementer ladning [C]
        self.eps0   = 8.854187817 * 10 ** (-12)     # [F/m]
        self.k      = 1.38064852 * 10 ** (-23)      # Boltzmanns konstant [J/K]
        self.h      = 6.62607015 * 10 ** (-34)      # Plancks konstant [J*s]
        self.N_A    = 6.022140857 * 10 ** 23        # Avogadros tall

        self.m_u    = m_u                           # Definerer for senere bruk
        self.T      = T                             # Temp til solkjerne [K]
        self.rho    = rho                           # Tettheten til kjernen [kg/m^3]

        self.X      = 0.7                           # Massebørk av hydrogen i kjernen til sola
        self.Y      = 0.29                          # Massebrøk av helium 4
        self.Y_3    = 10 ** (-10)                   # Massebrøk av helium 3
        self.Z_Li   = 10 ** (-7)                    # Massebrøk av lithium 7
        self.Z_Be   = 10 ** (-7)                    # Massebrøk av berylium 7
        self.Z_N    = 10 ** (-11)                   # Massebrøk av nitrogen 14

        self.N_H    = rho * self.X / ( m_u )        # Antalltettheten til hydrogen
        self.N_He4  = rho * self.Y / (4 * m_u)      # Antalltettheten til helium 4
        self.N_He3  = rho * self.Y_3 / (3 * m_u)    # Antalltettheten til helium 3
        self.N_Be   = rho * self.Z_Be / (7 * m_u)   # Antalltettheten til berylium
        self.N_Li   = rho * self.Z_Li / (7 * m_u)   # Antalltettheten til lithium
        self.N_N    = rho * self.Z_N / (14 * m_u)   # Antalltettheten til nitrogen
        self.N_e    = self.N_H + 2 * (self.N_He3 + self.N_He4) + 3 * self.N_Li + 4 * self.N_Be + 7 * self.N_N   # Antalltettheten til elektronet

        self.r_     = zeros(7)                      # [1/(kg*s)]
        self.Q      = zeros(7)                      # [J]


        # Spørr om sanitytest skal slås av eller på hvis dette ikke er prebestemt #
        while sanity != "J" and sanity != "N":
            sanity = input("Sanity test slått på? [J/N]: ")

        self.sanity = sanity

    def _Sanitytest(self):
        """ Printer testverdier som skal være kjente """

        # Henter rate og energi lister #
        r_  = self.r_
        Q   = self.Q

        tol = 1     # Toleransen for testene

        # Forventede verdier #
        exp_1   = 4.04 * 10 ** 2
        exp_2   = 8.68 * 10 ** (-9)
        exp_3   = 4.86 * 10 ** (-5)
        exp_4   = 1.49 * 10 ** (-6)
        exp_5   = 5.29 * 10 ** (-4)
        exp_6   = 1.63 * 10 ** (-6)
        exp_7   = 9.18 * 10 ** (-8)

        # Regner ut testresultater #
        res_1   = r_[0] * Q[0] * self.rho
        res_2   = r_[1] * Q[1] * self.rho
        res_3   = r_[2] * Q[2] * self.rho
        res_4   = r_[3] * Q[3] * self.rho
        res_5   = r_[4] * Q[4] * self.rho
        res_6   = r_[5] * Q[5] * self.rho
        res_7   = r_[6] * Q[6] * self.rho

        # Printer testverdier for å sammenlikne med forventede verdier #
        print("Prints values of sanity check")
        print("|Results                                     |Expected Values|")
        print("|r_HH     * (Q_HH + Q_HD)     * rho: %.2e|%15.2e|" %(res_1, exp_1))
        print("|r_He3He3 * Q_He3He3          * rho: %.2e|%15.2e|" %(res_2, exp_2))
        print("|r_He3He4 * Q_He3He4          * rho: %.2e|%15.2e|" %(res_3, exp_3))
        print("|r_Bee    * Q_Bee             * rho: %.2e|%15.2e|" %(res_4, exp_4))
        print("|r_LiH    * Q_LiH             * rho: %.2e|%15.2e|" %(res_5, exp_5))
        print("|r_BeH    * (Q_BeH + Q_decay) * rho: %.2e|%15.2e|" %(res_6, exp_6))
        print("|r_NH     * Q_CNO             * rho: %.2e|%15.2e|" %(res_7, exp_7))


    def Kalk(self):
        """ Regner ut energiproduksjonen fra hver av PP kjedene og de viktigste CNO kjedene for Sola """

        # Henter ut klassevariabler #
        rho     = self.rho
        elem    = self.elem
        eps0    = self.eps0
        k       = self.k
        h       = self.h
        T       = self.T
        T_9_1     = T * 10 ** (-9)                              # Skalert temperatur nummer 1
        T_9_2   = T_9_1 / ( 1 + 4.95 * 10 ** (-2) * T_9_1 )     # Skalert temperatur nummer 2
        T_9_3   = T_9_1 / ( 1 + 0.759 * T_9_1 )                 # Skalert temperatur nummer 3

        N_H     = self.N_H
        N_He4   = self.N_He4
        N_He3   = self.N_He3
        N_Be    = self.N_Be
        N_Li    = self.N_Li
        N_N     = self.N_N
        N_e     = self.N_e

        r_  = self.r_
        Q   = self.Q

        # Regner ut de forskjellige reaksjonsratene for prosessene #
        # Alle ratene er delt på Avogadros tall og ganget med      #
        # 10^-6 for å gjøre om til lambda_ik gitt i SI-enheter     #
        # Enheten for lambda = [s*K^(3/2)/(kg*m)]                  #
        lam_HH      = 4.01 * 10 ** (-15) * T_9_1 ** (- 2 / 3) * exp(-3.380 * T_9_1 ** (-1 / 3)) \
                      * (1 + 0.123 * T_9_1 ** (1 / 3) + 1.09 * T_9_1 ** (2 / 3) + 0.938 * T_9_1) / (self.N_A) * 10 ** (-6)

        lam_He3He3  = 6.04 * 10 ** 10 * T_9_1 ** (- 2 / 3) * exp(-12.276 * T_9_1 ** (-1 / 3)) \
                      * ( 1 + 0.034 * T_9_1 ** (1 / 3) - 0.522 * T_9_1 ** (2 / 3) - 0.124 * T_9_1 \
                      + 0.353 * T_9_1 ** (4 / 3) + 0.213 * T_9_1 ** (5 / 3) ) \
                      / (self.N_A) * 10 ** (-6)

        lam_He3He4  = 5.61 * 10 ** 6 * T_9_2 ** (5 / 6) * T_9_1 ** (- 3 / 2) * exp(-12.826 * T_9_2 ** (-1 / 3)) \
                      / (self.N_A) * 10 ** (-6)

        lam_Bee    = 1.34 * 10 ** (-10) * T_9_1 ** (- 1 / 2) \
                      * ( 1 - 0.537 * T_9_1 ** (1 / 3) + 3.86 * T_9_1 ** (2 / 3) \
                      + 0.0027 * T_9_1 ** (-1) * exp(2.515 * 10 ** (-3) * T_9_1 ** (-1)) ) \
                      / (self.N_A) * 10 ** (-6)

        lam_LiH     = ( 1.096 * 10 ** 9 * T_9_1 ** (-2 / 3) * exp(-8.472 * T_9_1 ** (-1 / 3)) \
                      - 4.830 * 10 ** 8 * T_9_3 ** (5 / 6) * T_9_1 ** (-3 / 2) * exp(-8.472 * T_9_3 ** (-1 / 3)) \
                      + 1.06 * 10 ** 10 * T_9_1 ** (-3 / 2) * exp(-30.442 * T_9_1 ** (-1)) ) \
                      / (self.N_A) * 10 ** (-6)

        lam_BeH    = ( 3.11 * 10 ** 5 * T_9_1 ** (-2 / 3) * exp(-10.262 * T_9_1 ** (-1 / 3)) \
                      + 2.53 * 10 ** 3 * T_9_1 ** (- 3 / 2) * exp(-7.306 * T_9_1 ** (-1)) ) \
                      / (self.N_A) * 10 ** (-6)

        lam_NH     = ( 4.90 * 10 ** 7 * T_9_1 ** (-2 / 3) * exp(-15.228 * T_9_1 ** (-1 / 3) - 0.092 * T_9_1 ** 2) \
                      * (1 + 0.027 * T_9_1 ** (1 / 3) - 0.778 * T_9_1 ** (2 / 3) - 0.149 * T_9_1 + 0.261 * T_9_1 ** (4 / 3) + 0.127 * T_9_1 ** (5 / 3)) \
                      + 2.37 * 10 ** 3 * T_9_1 ** (-3 / 2) * exp(-3.011 * T_9_1 ** (-1)) + 2.19 * 10 ** 4 * exp(-12.53 * T_9_1 ** (-1)) ) \
                      / (self.N_A) * 10 ** (-6)

        # Legger inn en grense for raten til berylium og elektronskollisjonene hvis temp er for lav #
        if lam_Bee > 1.57 * 10 ** (-7) * 10 ** (-6) / ( N_e * 6.022140857 * 10 ** 23 ) and T < 10 ** 6:
            lam_Bee = 1.57 * 10 ** (-7) * 10 ** (-6) / ( N_e * 6.022140857 * 10 ** 23 )

        # Regner ut reaksjonsraten per masse #
        r_[0]   = N_H ** 2 * lam_HH / ( 2 * rho )          # 0 = H   + H
        r_[1]   = N_He3 ** 2 * lam_He3He3 / ( 2 * rho )    # 1 = He3 + He3
        r_[2]   = N_He3 * N_He4 * lam_He3He4 / ( rho )     # 2 = He3 + He4
        r_[3]   = N_Be * N_e * lam_Bee / ( rho )           # 3 = Be  + e
        r_[4]   = N_Li * N_H * lam_LiH / ( rho )           # 4 = Li  + H
        r_[5]   = N_Be * N_H * lam_BeH / ( rho )           # 5 = Be  + H
        r_[6]   = N_N * N_H * lam_NH / ( rho )             # 6 = N   + H

        # Setter grense for at ikke en type reaksjon kan bruke mer enn andre reaksjoner har produsert #
        if r_[0] < 2 * r_[1] + r_[2]:           # Redigerer produksjonen fra helium reaksjonene hvis den er større enn helium produksjonen tillater
            C = r_[0] / (2 * r_[1] + r_[2])
            r_[1] = C * r_[1]
            r_[2] = C * r_[2]

        if r_[2] < r_[3] + r_[5]:               # Redigerer berylium reaksjonene etter helium reaksjonenes produksjon
            E = r_[2] / (r_[3] + r_[5])
            r_[3] = E * r_[3]
            r_[5] = E * r_[5]

        if r_[3] < r_[4]:                       # Redigerer lithium reaksjonen
            D = r_[3] / r_[4]
            r_[4] = D * r_[4]

        # Finner energien som blir produsert i de forskjellige reaksjonene #
        Q[0] = 1.177 + 5.494
        Q[1] = 12.86
        Q[2] = 1.586
        Q[3] = 0.049
        Q[4] = 17.346
        Q[5] = 0.137 + 1.022 + 7.345 + 2.995
        Q[6] = 1.944 + 1.513 + 7.551 + 7.297 + 1.757 + 4.966
        Q    = Q * 1.6021766 * 10 ** (-13)      # Gjør om fra MeV til J

        # Redefinerer klassevariablene #
        self.r_ = r_
        self.Q  = Q

        # Kjører tester hvis dette er slått på #
        if self.sanity == "J":
            self._Sanitytest()

        return self._Energiproduksjon()[4]

if __name__ == "__main__":
    A = Energy(1.57 * 10 ** 7, 1.62 * 10 ** 5)
    A.Kalk()
