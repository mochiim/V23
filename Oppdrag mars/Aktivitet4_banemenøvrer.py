import numpy as np

r = 149600000        # radius of Earth's orbit [km]
R = 227900000        # radius of Mars' orbit [km]

def Δv_1(r, R):
    """ Boost at Earth's orbit """
    Δv_1 = 29.78 * np.sqrt( (2*R) / (R + r) ) - 29.78
    return Δv_1

def Δv_2(r, R):
    """ Boost at Mars' orbit """
    Δv_1 = 24.13 - ( 24.14 * np.sqrt( (2*R) / (R + r)))
    return Δv_1

print(Δv_1(r, R))
print(Δv_2(r, R))
