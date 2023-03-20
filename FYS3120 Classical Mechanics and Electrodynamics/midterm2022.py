# Simple 1D pendulum example
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

# defining constants
g = sc.G

# defining constants
m1 = .16 # mass 1 of pendulum in kg
m2 = .01 # mass 2 of pendulum in kg

r1 = .02 # [m]
r2 = .16 # m

MR = m1*r1 - m2*r2
I = m1*r1**2 + m2*r2**2
k = 0.003

# determining the hamiltonian
def Hamiltonian(theta, p):
    return p**2/I - MR*g*np.cos(theta)

# Defining linspaces for coordinate values & conjugate momentum values
q = np.linspace(-2*np.pi, 2*np.pi, 101)
p = np.linspace(-10, 10, 101)
# Creating a meshgrid of the coordinate & conjugate momentum values
X, Y = np.meshgrid(q, p)

# Computing derivatives
u, v = Hamiltons_Equations(np.array([X, Y]))
# Making a stream plot of the Hamiltonian
fig, ax = plt.subplots()
strm = ax.streamplot(X, Y, u, v, linewidth=1)
plt.xlabel(r'$q(t)$', fontsize=10)
plt.ylabel(r'$p(t)$', fontsize=10)
plt.tight_layout()
#fig.savefig(f'ham_phasespace.pdf')

plt.show()
