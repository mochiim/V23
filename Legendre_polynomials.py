import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from math import factorial

plt.style.use('ggplot')
x = sym.Symbol('x')

def even_solution(m):
    """
    Independent solution for a finite sum given by a polynomial of degree l = 2m (even)
    """
    factor = factorial(m)**2 / factorial(2*m)
    sum = 0

    for n in range(m):
        sum += (-1)**n * x**(2*n) * ( factorial(2*m + 2*n) / (factorial(m + n)*factorial(m - n)*factorial(2*n)) )

    Q = factor*sum
    return Q

print(even_solution(2))


def odd_solution(m):
    """
    Independent solution for a finite sum given by a polynomial of degree l = 2m + 1 (odd)
    """
    factor = factorial(m)**2 / factorial(2*m + 1)
    sum = 0
    for n in range(m):
        sum += (-1)**n * x**(2*n + 1) * (factorial(2*m + 2*n + 1)) / (factorial(m + n)*factorial(m - n)*factorial(2*n + 1))

    Q = factor*sum
    return Q

#print(odd_solution(3))
