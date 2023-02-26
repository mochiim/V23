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
PP0 = np.zeros(2)
PP0[0] = energy_output(2*m_H, m_D2) - 0.265
PP0[1] = energy_output(m_D2 + m_H, m_He3)

PP1 = energy_output(2*m_He3, m_He4 + 2*m_H)

PP2 = np.zeros(3)
PP2[0] = energy_output(m_He3 + m_He4, m_Be7)
PP2[1] = energy_output(m_Be7 + m_e, m_Li7) - 0.815
PP2[2] = energy_output(m_Li7 + m_H, 2*m_He4)

PP3 = np.zeros(4)
PP3[0] = energy_output(m_He3 + m_He4, m_Be7)
PP3[1] = energy_output(m_Be7 + m_H, m_B8)
PP3[2] = energy_output(m_B8, m_Be8) - 6.711
PP3[3] = energy_output(m_Be8, 2*m_He4)


CNO = np.zeros(6)
CNO[0] = energy_output(m_C12 + m_H, m_N13)
CNO[1] = energy_output(m_N13, m_C13) - 0.707
CNO[2] = energy_output(m_C13 + m_H, m_N14)
CNO[3] = energy_output(m_N14 + m_H, m_O15)
CNO[4] = energy_output(m_O15, m_N15) - 0.997
CNO[5] = energy_output(m_N15 + m_H, m_C12 + m_He4)

print(CNO)
