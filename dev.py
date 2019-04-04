import functools as func
import tensorflow as tf

A = func.partial(tf.losses.huber_loss)

A()