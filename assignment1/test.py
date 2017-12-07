import numpy as np

a = np.array(range(12)).reshape(3,4)
print (a)

b = np.argmax(a, axis=1)
print (b)

