import numpy as np

a = np.array(range(12)).reshape(3,4)

a [:,2] = -1
print (a)

a[a>0] = 1
a[a<0] = 0
a[(1,2),(1,2)] = [3,2]
print (a)