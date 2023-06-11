import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ast

""" Task d """
hbar = ast.hbar.value                 # reduced Planck constant [J/s]
G = ast.G.value                       # gravitational constant [m^3/ (kg s^2)]
c = ast.c.value                       # speed of light in vacuum [m/s]

E_P = np.sqrt((hbar * c ** 5) / G)    # Planck energy

e_folds = 500
phi_init = E_P * np.sqrt((e_folds + 1/2 ) / (2*np.pi))    # initial value for the field                                 

""" Task e"""

_psi = lambda phi: phi/E_P
_dv_dpsi = lambda psi: 3 * hbar * c ** 5 / (4 * np.pi * G) * psi / phi_init ** 2
_v = lambda psi: 3/(8*np.pi) * (psi / _psi(phi_init))**2

def _solver(n = int(1e5)):
    """
    
    """
    # empty arrays for storing calculated values
    h = np.zeros(n)
    psi = np.zeros(n)
    dpsi = np.zeros(n)
    ln_a_ai = np.zeros(n)
    p_rho_c = np.zeros(n)

    # defining steplength and time interval
    tau_end = int(1e4) # arbitrary value
    dtau = tau_end / n
    tau = np.linspace(0, tau_end, n)

    # initial conditions
    h[0] = 1 # dimensionless Hubble parameter
    psi[0] = _psi(phi_init) # scalar field
    dpsi[0] = 0 # derivative of scalar field
    ln_a_ai[0] = 0 # logarithm of scale factor
    p_rho_c[0] = (.5 * dpsi[0] ** 2 - _v(psi[0])) / (.5 * dpsi[0] ** 2 + _v(psi[0])) # ratio between p and ρc^2

    for i in range(n - 1):
        d2psi = - 3 * h[i] * dpsi[i] - _dv_dpsi(psi[i]) # double derivative of scalar field
        dpsi[i+1] = dpsi[i] + d2psi * dtau
        psi[i+1] = psi[i] + dpsi[i] * dtau
        h[i+1] = np.sqrt( np.abs( 8 * np.pi / 3 * (0.5 * dpsi[i + 1] ** 2 + _v(psi[i + 1])) ))
        ln_a_ai[i + 1] = ln_a_ai[i] + h[i] * dtau
        p_rho_c[i + 1] = (.5 * dpsi[i + 1] ** 2 - _v(psi[i + 1])) / (.5 * dpsi[i + 1] ** 2 + _v(psi[i + 1]))

    return h, tau, psi, dpsi, ln_a_ai, p_rho_c

def _taske(plot = False, save = False):
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()

    if plot:
        fig, ax = plt.subplots(2, sharex=True, figsize = (6, 6))
        ax[0].plot(tau, psi/_psi(phi_init), color = "black")
        ax[0].set_ylabel(r"$\psi / \psi_i$")
        ax[0].set_xlim([0, 2000])

        ax[1].plot(tau, ln_a_ai, color = "black")
        ax[1].set_ylabel(r"ln(a/a$_i$)")
        ax[1].set_xlabel(r"$\tau$")
        ax[1].set_xlim([0, 2000])

        # create a zoom in window
        axins = ax[0].inset_axes([.75, .2, .2, .7])
        axins.plot(tau, psi/_psi(phi_init), color = "black")
        x1, x2, y1, y2 = 1010, 1410, -0.02, .02
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.tick_params(labelbottom = False, labelleft = False)
        ax[0].indicate_inset_zoom(axins, edgecolor = "black")

    if save:
        plt.savefig("task_e.png")

""" Task f"""

def _taskf(plot = False, save = False):
    """
    SRA
    """
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()
    SRA_psi = _psi(phi_init) - tau / (4 * np.pi *_psi(phi_init))
    
    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(tau, psi, color = "black", label = "Numerical solution")
        plt.plot(tau, SRA_psi, label = "Slow-roll approximation", color = "black", ls = "dashed")
        plt.xlim([0, 2000]); plt.ylim([-3, 4])
        plt.xlabel(r"$\tau$"), plt.ylabel(r"$\psi$")
        plt.legend()

    if save:
        plt.savefig("task_f.png")

""" Task g"""
_epsilon = lambda phi: E_P ** 2 / (4 * np.pi * phi ** 2)

def _taskg(plot = False, save = False):
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()
    phi = psi * E_P
    epsilon = _epsilon(phi)

    end_of_inflation = np.where(epsilon >= 1)[0][0] # finding index for where inflation ends, i.e. epsilon = 1
    N_tot = ln_a_ai[end_of_inflation] # number of e-folds
    print(f"Numerical N_tot: {N_tot}")

    if plot:
        plt.figure(figsize = (6, 6))
        plt.plot(tau, epsilon, color = "black")
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\epsilon$")
        plt.yscale("log")
        plt.axhline(1, color = "black", ls = "dashed") # indicating end of inflation
        plt.xlim([0, 1500]); plt.ylim([0, 1e10])

    if save:
        plt.savefig("task_g.png")

    return N_tot

""" Task i """
def _taski(plot = False, save = False):
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()

    if plot:
        plt.figure(figsize = (6, 6))
        plt.plot(tau, p_rho_c, color = "black")
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$p_\phi / \rho_\phi c^2$")
        plt.xlim([0, 1500])
    
    if save: 
        plt.savefig("task_i.png")

""" Task j"""
def _taskj(plot = False, save = False):
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()
    phi = psi * E_P
    N_tot = _taskg(plot = False, save = False)
    epsilon = _epsilon(phi)
    idx = epsilon <= 1
    N_left = N_tot - ln_a_ai[idx]
    
    if plot:
        plt.figure(figsize = (6, 6))
        plt.plot(N_left, epsilon[idx], color = "black")
        plt.xscale("log")
        plt.xlabel("N"); plt.ylabel(r"$\epsilon$")
    
    if save:
        plt.savefig("task_j.png")

if __name__ == "__main__":
    #_taske(plot = False, save = False)
    #_taskf(plot = False, save = False)
    #_taskg(plot = False, save = False)
    #_taski(plot = False, save = False)
    _taskj(plot = True, save = True)
    plt.show()
 
    
