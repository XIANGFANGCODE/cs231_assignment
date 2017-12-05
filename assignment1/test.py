import numpy as np

a = np.array(range(12)).reshape(3,4)
b = np.random.rand()

print (a)
print (np.mean(a, axis=0))
print (np.sum(a,axis=0))