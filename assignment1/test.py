#import numpy as np
import tensorflow as tf
import numpy as np
import math
import timeit

a = np.random.randn(2,3,4)
print(a)

x_shape = tf.get_shape
print(x_shape)