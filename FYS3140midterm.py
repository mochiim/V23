import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

t = np.linspace(0, 10, 1000)

x = lambda k: - ((k*sc.c) /(np.sqrt(1 - k**2)))*t
plt.plot(t, x(.3))
plt.plot(t, x(.5))
plt.plot(t, x(.005))
plt.show()
