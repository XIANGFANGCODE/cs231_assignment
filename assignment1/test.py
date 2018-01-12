#import numpy as np
import tensorflow as tf
import numpy as np
import math
import timeit

c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(c)
s = c.shape
print(c.shape)
l = s.as_list()
print(l)