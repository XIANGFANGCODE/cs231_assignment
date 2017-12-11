import numpy as np

a = np.array(range(24)).reshape(2,3,4)
print (a)
print (a[...,:2])

b = np.ones(2)
print (a[...,:2].shape)
print (b)
print (b.shape)
c = np.dot(a[...,:2],b)
print (c)
print (c.size)