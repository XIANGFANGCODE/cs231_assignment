#import numpy as np
import tensorflow as tf
import numpy as np
import math
import timeit

f = np.random.randn(2,3)
print(f)

print(np.argmax(f, axis=1))