import numpy as np

A = np.arange(95,99).reshape(2,2)
print (A)
pad = 1
A = np.pad(A,((pad,pad),(pad,pad)),'constant',constant_values = (0,0))
print (A)

print (A[:,:])

