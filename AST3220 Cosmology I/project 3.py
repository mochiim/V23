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
_v = lambda psi: (3 * hbar * c ** 5) / (8 *np.pi * G) * (psi / phi_init) ** 2

def _solver(n = int(1e5)):

    h = np.zeros(n)
    psi = np.zeros(n)
    dpsi = np.zeros(n)
    ln_a_ai = np.zeros(n)

    # defining steplength and time interval
    tau_end = int(1e4) # arbitrary value
    dtau = tau_end / n
    tau = np.linspace(0, tau_end, n)

    # initial conditions
    h[0] = 1
    psi[0] = _psi(phi_init)
    dpsi[0] = 0
    ln_a_ai[0] = 0

    for i in range(n - 1):
        d2psi = - 3 * h[i] * dpsi[i] - _dv_dpsi(psi[i]) # double derivative of scalar field

        dpsi[i+1] = dpsi[i] + d2psi * dtau

        psi[i+1] = psi[i] + dpsi[i] * dtau

        h[i+1] = np.sqrt( np.abs( 8 * np.pi / 3 * (0.5 * dpsi[i + 1] ** 2 + _v(psi[i + 1])) ))

        ln_a_ai[i + 1] = ln_a_ai[i] + h[i] * dtau

    return h, tau, psi, dpsi, ln_a_ai

def _taske(save = False):
    h, tau, psi, dpsi, ln_a_ai = _solver()
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

def _taskf(save = False):
    """
    SRA
    """
    h, tau, psi, dpsi, ln_a_ai = _solver()
    SRA_psi = _psi(phi_init) - tau / (4 * np.pi *_psi(phi_init))
    plt.figure(figsize=(6, 6))
    plt.plot(tau, psi, color = "black", label = "Numerical solution")
    plt.plot(tau, SRA_psi, label = "Slow-roll approximation", color = "black", ls = "dashed")
    plt.xlim([0, 2000]); plt.ylim([-3, 4])
    plt.xlabel(r"$\tau$"), plt.ylabel(r"$\psi$")
    plt.legend()

    if save:
        plt.savefig("task_f.png")

if __name__ == "__main__":
    #_taske(save = False)
    #_taskf(save = False)
    plt.show()
 
    
