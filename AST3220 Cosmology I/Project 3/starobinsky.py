import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ast

hbar = ast.hbar.value                 # reduced Planck constant [J/s]
G = ast.G.value                       # gravitational constant [m^3/ (kg s^2)]
c = ast.c.value                       # speed of light in vacuum [m/s]
E_P = np.sqrt(1 / G)                  # Planck energy
M_P = np.sqrt(1 / (8 * np.pi * G)) # Planck mass

# initial conditions
psi_i = 2                             # initial condition of the field
phi_i = psi_i * E_P
y_i = - np.sqrt(16 * np.pi / 3) * psi_i

# useful functions
_psi = lambda phi: phi / E_P
_y = lambda psi: - np.sqrt(16 * np.pi / 3) * psi
_v = lambda psi: 3 / (8 * np.pi) *  (1 - np.exp(_y(psi))) ** 2  / (1 - np.exp(y_i)) ** 2
_dv_dpsi = lambda psi: np.sqrt(3 / np.pi) * np.exp(_y(psi)) * (1 - np.exp(_y(psi))) / (1 - np.exp(y_i)) ** 2

def _plot(x, y, xlabel, ylabel):
    """
    Plotting function

    Arguments:
    x: array to be plotted on the x-axis
    y: array to be plotted on the y-axis
    xlabel: string for label on x-axis
    ylabel: string for label on y-axis
    """
    plt.figure(figsize = (6, 6))
    plt.plot(x, y, color = "black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


""" Task m """
def _solver(n = int(1e6)):

    h = np.zeros(n)
    psi = np.zeros(n)
    dpsi = np.zeros(n)
    ln_a_ai = np.zeros(n)
    p_rho_c = np.zeros(n)


    tau_end = int(1e4) # arbitrary value
    dtau = tau_end / n
    tau = np.linspace(0, tau_end, n)

    # initial conditions
    h[0] = 1 
    psi[0] = psi_i
    dpsi[0] = 0 
    ln_a_ai[0] = 0
    p_rho_c[0] = (.5 * dpsi[0] ** 2 - _v(psi[0])) / (.5 * dpsi[0] ** 2 + _v(psi[0])) 

    for i in range(n - 1):
        d2psi = - 3 * h[i] * dpsi[i] - _dv_dpsi(psi[i]) 
        dpsi[i+1] = dpsi[i] + d2psi * dtau
        psi[i+1] = psi[i] + dpsi[i] * dtau
        h[i+1] = np.sqrt( np.abs( 8 * np.pi / 3 * (0.5 * dpsi[i + 1] ** 2 + _v(psi[i + 1])) ))
        ln_a_ai[i + 1] = ln_a_ai[i] + h[i] * dtau
        p_rho_c[i + 1] = (.5 * dpsi[i + 1] ** 2 - _v(psi[i + 1])) / (.5 * dpsi[i + 1] ** 2 + _v(psi[i + 1]))

    return h, tau, psi, dpsi, ln_a_ai, p_rho_c

def _taskm(plot = False, save = False):
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()

    if plot:
        fig, ax = plt.subplots(2, sharex=True, figsize = (6, 6))
        ax[0].plot(tau, psi, color = "black")
        ax[0].set_ylabel(r"$\psi / \psi_i$")
        ax[0].set_xlim([0, 5000])

        ax[1].plot(tau, ln_a_ai, color = "black")
        ax[1].set_ylabel(r"ln(a/a$_i$)")
        ax[1].set_xlabel(r"$\tau$")
        ax[1].set_xlim([0, 5000])

        axins = ax[0].inset_axes([.75, .2, .2, .7])
        axins.plot(tau, psi, color = "black")
        x1, x2, y1, y2 = 2661, 3199, -0.02, .02
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.tick_params(labelbottom = False, labelleft = False)
        ax[0].indicate_inset_zoom(axins, edgecolor = "black")
    
    if save:
        plt.savefig("task_m.png")

""" Task n """
# slow-roll paramters
_epsilon = lambda psi: 4 / 3 * np.exp(2 * _y(psi)) / (1 - np.exp(_y(psi))) ** 2
_eta = lambda psi: (4/3) * (2 * np.exp(2 * _y(psi)) - np.exp(_y(psi))) / (1 - np.exp(_y(psi))) ** 2

def _taskn(plot = False, save = False):
    h, tau, psi, dpsi, ln_a_ai, p_rho_c = _solver()

    # task n: repeat of steps j)
    eps = _epsilon(psi)
    eta = _eta(psi)
    end_of_inflation = np.where(eps >= 1)[0][0] 
    N_tot = ln_a_ai[end_of_inflation] 
    idx = eps <= 1
    N_left = N_tot - ln_a_ai[idx]

    # task o
    N_approx = (3 / 4) * np.exp(-_y(psi))
    eps_approx = 3 / (4 * N_approx ** 2)
    eta_approx = - 1 / N_approx
    idx_approx = eps_approx <= 1

    # task n: repeat of steps k)
    tau_end = tau[np.where(eps >= 1)][0]
    psi_end = psi[np.where(tau == tau_end)][0]
    phi_end = psi_end * E_P

    psi_idx = []
    for i in range(len(N)):
        if N[i] > 50 and N[i] < 60:
            eps_idx.append(psi[i])

    n = 1 - 6 * _epsilon(np.array(psi_idx)) + 2 * _eta(np.array(psi_idx))
    r = 16 * _epsilon(np.array(psi_idx))

    plt.plot(n, r)

    if plot:
        plt.plot(N_left, eps[idx], color = "black", ls = "solid", label = r"$\epsilon_{numerical}$")
        plt.plot(N_left, eta[idx], color = "red", ls = "solid", label = r"$\eta_{numerical}$")
        plt.plot(N_approx[idx_approx], eps_approx[idx_approx], color = "black", ls = "dashed", label = r"$\epsilon_{approximated}$")
        plt.plot(N_approx[idx_approx], eta_approx[idx_approx], color = "red", ls = "dashed", label = r"$\eta_{approximated}$")
        plt.xlabel("N"); plt.ylabel(r"$\epsilon, \eta$")
        plt.xscale("log")
        plt.legend()

    if save: 
        plt.savefig("task_o_and_n_1.png")
    
if __name__ == "__main__":
    _taskm(plot = False, save = False)
    _taskn(plot = False, save = False)
    plt.show()