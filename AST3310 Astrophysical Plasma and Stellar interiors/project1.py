import numpy as np
import matplotlib.pyplot as pyplot

class energy:
    """
    A class which accepts a temperature and density, and then calculates the
    amount of energy produced by the fusion chains discussed in Chapter 3
    """
    def __init__(self, T, rho):

        self.mu = 1.6606*1e-27  # atomic mass unit [kg]
        self.c = 2.9979*1e8     # sped of light [m/s]
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
        self.Q = np.zeros(6)            # energy released per reaction [J]
        self.Q[0] = 1.177 + 5.494            # PP0 [MeV]
        self.Q[1] = 12.860                   # PP1 [MeV]
        self.Q[2] = 1.586                    # PPII and PPIII [MeV]
        self.Q[3] = 0.049 + 17.346           # PPII [MeV]
        self.Q[4] = 0.137 + 8.367 + 2.995    # PPIII [MeV]
        self.Q[5] = (1.944 + 1.513 + 7.551 + \
                7.297 + 1.757 + 4.966)  # CNO cycle [MeV]
        self.Q = self.Q*(self.eV*1e6)          # [MeV] -> [J]

energy = energy(1, 2)
print(energy.Q)

# Temperature of solar core T = 1.57*1e7 [K]
# Density of solar core rho = 1.62*1e5 [kg/m^3]
# Adjusted temperature T = 1e8 [K]
