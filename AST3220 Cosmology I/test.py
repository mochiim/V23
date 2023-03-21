import numpy as np
import matplotlib.pyplot as plt

steps = 1000
N = np.linspace(-np.log(1 + 2e7), 0, steps)
idx = np.where(N == -np.log(3))[0][0]
