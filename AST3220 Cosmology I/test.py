import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const

h_bar = const.hbar.value
c = const.c.value
G = const.G.value

Ep = np.sqrt(h_bar * c**5 / G)
Mp = np.sqrt(h_bar * c / G)
phi_i = np.sqrt( Ep**2/(2 * np.pi) * (500 + 1/2) )
psi_i = phi_i / Ep

def psi_func(phi):
    return phi / Ep

def v_func(psi):
    return 3 * h_bar * c**5 / (8 * np.pi * G) * psi**2 / phi_i**2

def dv_dpsi_func(psi):
    return 3 * h_bar * c**5 / (4 * np.pi * G) * psi / phi_i**2

def solve(n=int(1e6)):
    """
    Solve equations 9, 10 and 11 numerically.
    Parameters:
        n (int): iterations (100 000 by default)
    """

    tau_end = int(3e4)
    dt = tau_end / n

    h = np.zeros(n)
    psi = np.zeros(n)
    d_psi = np.zeros(n)
    p_rhoc = np.zeros(n)

    tau = np.linspace(0, tau_end, n)

    psi[0] = psi_func(phi_i)
    d_psi[0] = 0
    h[0] = 1

    p_rhoc[0] = (0.5 * d_psi[0]**2 - v_func(psi[0]) ) \
              / (0.5 * d_psi[0]**2 + v_func(psi[0]))

    for i in range(n-1):
        var = - 3 * h[i] * d_psi[i] - dv_dpsi_func(psi[i])

        d_psi[i+1] = d_psi[i] + var * dt

        psi[i+1] = psi[i] + d_psi[i] * dt

        v = v_func(psi[i+1])

        h[i+1] = np.sqrt(np.abs( 8 * np.pi / 3 * (0.5 * d_psi[i+1]**2 + v) ))

        p_rhoc[i+1] = (0.5 * d_psi[i+1]**2 - v) / (0.5 * d_psi[i+1]**2 + v)

    return psi, tau, h, p_rhoc, dt

psi, tau, h, p_rhoc, dt = solve()

def e():
    """Plot the numerical solution of eq. (9), (10) and (11)"""
    plt.plot(tau, psi)
    plt.xlim(0, 3000); plt.ylim(-1, 4)
    plt.xlabel(r'$\tau$', fontsize=15); plt.ylabel(r'$\psi$', fontsize=15)
    plt.grid(); plt.show()

    plt.plot(tau, psi)
    plt.title(r'Close-up of late oscillating phase for $\phi^2$-model')
    plt.xlim(15_000, 30_000); plt.ylim(-0.001, 0.001)
    plt.xlabel(r'$\tau$', fontsize=15); plt.ylabel(r'$\psi$', fontsize=15)
    plt.grid(); plt.show()

def f():
    """Plot the numerical and analytical solutions"""
    analytical = psi[0] - 1/ (4 * np.pi * psi[0]) * tau

    plt.plot(tau, psi, label='Numerical solution')
    plt.plot(tau, analytical, label='Analytical solution')
    plt.legend(fancybox=True); plt.grid()
    plt.xlim(600, 1600); plt.ylim(-4, 4)
    plt.xlabel(r'$\tau$', fontsize=15); plt.ylabel(r'$\psi$', fontsize=15)
    plt.show()

def epsilon(phi):
    return Ep**2 / (4 * np.pi * phi**2)

def g(plot=True):
    """Solve task g"""

    phi = psi * Ep
    eps = epsilon(phi)

    tau_end = tau[np.where(eps >= 1)][0]
    psi_end = psi[np.where(tau==tau_end)][0]
    phi_end = psi_end * Ep

    if plot:
        plt.yscale('log')
        plt.plot(tau, eps)
        plt.grid()
        plt.xlim(0, 2000); plt.ylim(0, 1e13)
        plt.xlabel(r'$\tau$', fontsize=15)
        plt.ylabel(r'$\epsilon$', fontsize=15)
        plt.show()

        print(f'\nNumerical N_tot: {N_tot(phi_end):.6f}\n')

    return tau_end, psi_end, phi_end

def i():
    """Solve task i"""
    plt.plot(tau, p_rhoc)
    plt.grid()
    plt.xlim(0, 2000)
    plt.xlabel(r'$\tau$', fontsize=15)
    plt.ylabel(r'$\frac{p_\phi}{\rho_\phi c^2}$', fontsize=15, rotation=0)
    plt.show()

def j(plot=True):
    """Solve task j"""
    ln_a_ai = np.zeros(len(h))
    ln_a_ai[0] = h[0]
    for i in range(len(h)-1):
        ln_a_ai[i+1] = ln_a_ai[i] + h[i+1]*dt

    ln_tp_mat = 2/3*np.log(tau) + ln_a_ai[-1] - 2/3*np.log(tau[-1])
    ln_tp_rad = 1/2*np.log(tau) + ln_a_ai[-1] - 1/2*np.log(tau[-1])

    if plot:
        plt.plot(np.log(tau), ln_a_ai, label=r'$\ln(a/a_i)$')
        plt.plot(np.log(tau), ln_tp_mat, label=r'$p = 2/3$')
        plt.plot(np.log(tau), ln_tp_rad, label=r'$p = 1/2$')
        plt.grid(); plt.legend(fancybox=True)
        plt.xlim(6, 11); plt.ylim(500, 514)
        plt.xlabel(r'$\tau$', fontsize=15)
        plt.ylabel(r'$\ln(a/a_i)$', fontsize=15)
        plt.show()

    return ln_a_ai

def N_tot(phi_end):
    return 2 * np.pi / Ep**2 * (phi_i**2 - phi_end**2)

tau_end, psi_end, phi_end = g(plot=False)

def k():
    """Solve task k"""
    ln_a_ai = j(plot=False)
    N = N_tot(phi_end) - ln_a_ai

    phi_N = np.sqrt(N * Ep**2 / (2 * np.pi) + phi_end**2)

    plt.plot(N, epsilon(phi_N))
    plt.xlim(0, 70); plt.ylim(-0.2, 1)
    plt.grid()
    plt.xlabel('N', fontsize=15)
    plt.ylabel(r'$\epsilon$', fontsize=15)
    plt.show()

def l():
    """Solve task l"""
    N = np.linspace(50, 60)
    phi_N = np.sqrt(N * Ep**2 / (2*np.pi) + phi_end**2 )
    eps_N = epsilon(phi_N)

    n = 1 - 4 * eps_N # SRA: eta = epsilon
    r = 16 * eps_N

    n_obs = 0.965

    plt.plot(n, r, label='numerical')
    plt.vlines(n_obs, 0.13, 0.16, 'k', label='observed', ls=':')
    plt.ylim(0.132, 0.158)
    plt.legend(fancybox=True, loc='lower left', framealpha=1)
    plt.xlabel('n'); plt.ylabel('r'); plt.grid()
    plt.show()



if __name__ == "__main__":

    e()
    f()
    g()
    i()
    j()
    k()
    l()
