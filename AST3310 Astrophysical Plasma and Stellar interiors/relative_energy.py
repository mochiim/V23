import numpy as np
import matplotlib.pyplot as plt
from energy_production import energy

plt.rcParams['font.size'] = 16

n = 10000                      # number of steps
rho = 1.62*1e5                # density of solar core [kg/m^3]
T = np.logspace(4, 9, n)      # temperature interval we are plotting over [K]
E_rel = np.zeros((4, n))      # relative energy production in a star

for i in range(n):
    star = energy(T[i], rho)            # creating an instance in class energy with given temperature and density
    star.reaction_rates()               # reaction rates calculated based on given temperature
    PP1, PP2, PP3, CNO = star.energy_production() # energy production
    E_tot = PP1 + PP2 + PP3 + CNO

    E_rel[0, i] = PP1/E_tot # PP1
    E_rel[1, i] = PP2/E_tot # PP2
    E_rel[2, i] = PP3/E_tot # PP3
    E_rel[3, i] = CNO/E_tot # CNO

plt.figure(figsize=(10, 6))
plt.plot(T, E_rel[0, :], label = "PP1")
plt.plot(T, E_rel[1, :], label = "PP2")
plt.plot(T, E_rel[2, :], label = "PP3")
plt.plot(T, E_rel[3, :], label = "CNO")
plt.xscale('log')
plt.ylabel("Relative energy")
plt.xlabel("Temperature [K]")
plt.title("Relative energy production")
plt.legend()
#plt.savefig("releng1.png")
plt.show()
