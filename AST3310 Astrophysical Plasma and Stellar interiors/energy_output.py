import numpy as np

# defining useful variables
u = 1.6606e-27                    # atomic mass unit [kg]
c = 2.9979e8                      # speed of light [ms^-1]
eV = 1.6022e-19                   # [J]
conversion = (u*c**2)/(eV*1e6)    # conversion factor from [u] -> [MeV]

#  atomic mass of different elements
m_p = 1.0073           # proton [u]
m_e = 5.4858e-4        # electron [u]
m_H = 1.007825         # Hydrogen [u]
m_D2 = 2.014           # Deuterium [u]
m_He3 = 3.0160         # Helium 3 [u]
m_He4 = 4.0026         # Helium 4[u]
m_Be7 = 7.01693		   # Beryllium 7[u]
m_Be8 = 8.00531		   # Beryllium 8 [u]
m_Li7 = 7.016004	   # Lithium 7 [u]
m_B8 = 8.02461		   # Boron 8 [u]

m_C12 = 12				# Carbon 12 mass [u]
m_C13 = 13.00336			# Carbon 13 mass [u]
m_N13 = 13.00574			# Nitrogen 13 mass [u]
m_N14 = 14.00307			# Nitrogen 14 mass [u]
m_N15 = 15.00109			# Nitrogen 15 mass [u]
m_O15 = 15.00307	        # Oxygen 15 [u]

def energy_output(initial_mass, final_mass):
    """
    A function which calculated the energy output [MeV] from a fusion reaction based
    on the initial mass [u] and final mass [u]
    """
    E = (initial_mass - final_mass) * conversion
    return E

# energy released in neutrinos [MeV]
Q_nu = np.array([2*0.265, 0.265  + 0.815, 0.265 + 6.711, 0.707 + 0.997])

# calulating energy output of PP0
PP0 = np.zeros(2)
PP0[0] = energy_output(2*m_H, m_D2) - 0.265         # neutrino energy [MeV] added
PP0[1] = energy_output(m_D2 + m_H, m_He3)

# calulating energy output of PP1
PP1 = energy_output(2*m_He3, m_He4 + 2*m_H)

# calulating energy output of PP2
PP2 = np.zeros(3)
PP2[0] = energy_output(m_He3 + m_He4, m_Be7)
PP2[1] = energy_output(m_Be7 + m_e, m_Li7) - 0.815  # neutrino energy [MeV] added
PP2[2] = energy_output(m_Li7 + m_H, 2*m_He4)

# calulating energy output of PP3
PP3 = np.zeros(4)
PP3[0] = energy_output(m_He3 + m_He4, m_Be7)
PP3[1] = energy_output(m_Be7 + m_H, m_B8)
PP3[2] = energy_output(m_B8, m_Be8) - 6.711         # neutrino energy [MeV] added
PP3[3] = energy_output(m_Be8, 2*m_He4)

# calulating energy output of CNO cycle
CNO = np.zeros(6)
CNO[0] = energy_output(m_C12 + m_H, m_N13)
CNO[1] = energy_output(m_N13, m_C13) - 0.707        # neutrino energy [MeV] added
CNO[2] = energy_output(m_C13 + m_H, m_N14)
CNO[3] = energy_output(m_N14 + m_H, m_O15)
CNO[4] = energy_output(m_O15, m_N15) - 0.997        # neutrino energy [MeV] added
CNO[5] = energy_output(m_N15 + m_H, m_C12 + m_He4)

# total energy output per fusion branch
e_tot = np.array([2*np.sum(PP0) + np.sum(PP1), np.sum(PP0) + np.sum(PP2), np.sum(PP0) + np.sum(PP3), np.sum(CNO)])

print("Branch | Energy output [MeV]  | Expected [MeV] | Neutrino energy [MeV] | Energy lost to neutrino energy ")
print(f"PP1    | {e_tot[0]: .5}              |    26.202      | {Q_nu[0]}                  | {Q_nu[0]/e_tot[0]: .3}%")
print(f"PP2    | {e_tot[1]: .5}              |    25.652      | {Q_nu[1]}                  | {Q_nu[1]/e_tot[1]: .3}%")
print(f"PP3    | {e_tot[2]: .5}              |    18.59       | {Q_nu[2]}                 | {Q_nu[2]/e_tot[2]: .3}%")
print(f"CNO    | {e_tot[3]: .5}               |    25.028      | {Q_nu[3]}                 | {Q_nu[3]/e_tot[3]: .3}%")

"""
Branch | Energy output [MeV]  | Expected [MeV] | Neutrino energy [MeV] | Energy lost to neutrino energy
PP1    |  26.204              |    26.202      | 0.53                  |  0.0202%
PP2    |  26.165              |    25.652      | 1.08                  |  0.0413%
PP3    |  19.758              |    18.59       | 6.976                 |  0.353%
CNO    |  25.03               |    25.028      | 1.704                 |  0.0681%
"""
