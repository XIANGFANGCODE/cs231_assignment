import numpy as np

a = np.array(range(4)).reshape(2,2)
print (a)


b=np.copy(a)

b[0,1] = 122
print(b)
print(a)