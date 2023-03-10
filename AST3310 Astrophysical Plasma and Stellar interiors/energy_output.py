import numpy as np
import scipy.constants as const

# defining useful variables
u = const.u                     # atomic mass unit [kg]
c = const.c # speed of light [ms^-1]
eV = const.eV                   # [J]
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

m_C12 = 12				    # Carbon 12 mass [u]
m_C13 = 13.00336			# Carbon 13 mass [u]
m_N13 = 13.00574			# Nitrogen 13 mass [u]
m_N14 = 14.00307			# Nitrogen 14 mass [u]
m_N15 = 15.00109			# Nitrogen 15 mass [u]
m_O15 = 15.00307	        # Oxygen 15 [u]

def energy_output(initial_mass, final_mass):
    """
    A function which calculated the energy output [MeV] from a fusion reaction based
    on difference in initial mass and final mass given in units of u and converts to MeV.
    """
    E = (initial_mass - final_mass) * conversion
    return E

# energy released in the reaction and neutrino energies in PP chain and CNO cycle [MeV]
PP_Q = np.zeros((10, 2))
PP_Q[0, :] = np.array([1.177, 0.265])    # neutrino emitted
PP_Q[1, :] = np.array([5.494, 0])
PP_Q[2, :] = np.array([12.860, 0])
PP_Q[3, :] = np.array([1.586, 0])
PP_Q[4, :] = np.array([0.049, 0.815])    # neutrino emitted
PP_Q[5, :] = np.array([17.346, 0])
PP_Q[6, :] = np.array([1.586, 0])
PP_Q[7, :] = np.array([0.137, 0])
PP_Q[8, :] = np.array([8.367, 6.711])    # neutrino emitted
PP_Q[9, :] = np.array([2.995, 0])

CNO_Q = np.zeros((7, 2))
CNO_Q[0] = np.array([1.944, 0])
CNO_Q[2] = np.array([1.513, 0.707])      # neutrino emitted
CNO_Q[3] = np.array([7.551, 0])
CNO_Q[4] = np.array([7.297, 0])
CNO_Q[5] = np.array([1.757, 0.997])      # neutrino emitted
CNO_Q[6] = np.array([4.966, 0])

# calulating energy output of PP0
PP0 = np.zeros(2)
PP0[0] = energy_output(2*m_H, m_D2)
PP0[1] = energy_output(m_D2 + m_H, m_He3)

# calulating energy output of PP1
PP1 = energy_output(2*m_He3, m_He4 + 2*m_H)

# calulating energy output of PP2
PP2 = np.zeros(3)
PP2[0] = energy_output(m_He3 + m_He4, m_Be7)
PP2[1] = energy_output(m_Be7, m_Li7)
PP2[2] = energy_output(m_Li7 + m_H, 2*m_He4)

# calulating energy output of PP3
PP3 = np.zeros(4)
PP3[0] = energy_output(m_He3 + m_He4, m_Be7)
PP3[1] = energy_output(m_Be7 + m_H, m_B8)
PP3[2] = energy_output(m_B8, m_Be8)
PP3[3] = energy_output(m_Be8, 2*m_He4)

# calulating energy output of CNO cycle
CNO = np.zeros(6)
CNO[0] = energy_output(m_C12 + m_H, m_N13)
CNO[1] = energy_output(m_N13, m_C13)
CNO[2] = energy_output(m_C13 + m_H, m_N14)
CNO[3] = energy_output(m_N14 + m_H, m_O15)
CNO[4] = energy_output(m_O15, m_N15)
CNO[5] = energy_output(m_N15 + m_H, m_C12 + m_He4)


pp0 = np.sum(PP_Q[:2, 0]) # energy output from PP0 (defined for convenience)

# energy output for all PPP branches and CNO cycle
E_PP1 = 2*np.sum(PP0) + np.sum(PP1)
E_PP2 = np.sum(PP0) + np.sum(PP2)
E_PP3 = np.sum(PP0) + np.sum(PP3)
E_CNO = np.sum(CNO)

# computing the neutrino energy
nu_PP1 = 2*pp0 + np.sum(PP_Q[2, 0])
nu_PP2 = pp0 + np.sum(PP_Q[3:6, 0])
nu_PP3 = pp0 + np.sum(PP_Q[6:, 0])
nu_CNO = np.sum(CNO_Q[:, 0])

names = ["PP1", "PP2", "PP3", "CNO"]
dE = [E_PP1, E_PP2, E_PP3, E_CNO]
neutrino = [E_PP1 - nu_PP1, E_PP2 - nu_PP2, E_PP3 - nu_PP3, E_CNO - nu_CNO]

print("Branch | Mass difference [MeV] | Neutrino energy [MeV] | Percentage loss ")
for i in range(4):
    print(f"{names[i]}    | {dE[i]: .3f}               | {neutrino[i]: .3f}                |  {neutrino[i]/dE[i]*100: .3f}%")

"""
Branch | Mass difference [MeV] | Neutrino energy [MeV] | Percentage loss
PP1    |  26.734               |  0.532                |   1.990%
PP2    |  26.734               |  1.082                |   4.047%
PP3    |  26.734               |  6.978                |   26.101%
CNO    |  26.734               |  1.706                |   6.381%
"""
