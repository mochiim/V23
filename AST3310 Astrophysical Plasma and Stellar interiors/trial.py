import numpy as np
import numpy.ma as ma

a = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 4, 5])
print(a)


plus = ma.masked_greater_equal(a, 0).mask * 2
neg = ma.masked_less(a, 0).mask * 1
#print(plus)
#print(neg)
#print(plus + neg)

b = np.array([0, 0, 1, 1, 0, 0])
print(ma.masked_where(b != 0, b).mask)
