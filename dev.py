import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(int(5)&1)

tf.enable_eager_execution()

print(tf.losses.sigmoid_cross_entropy(np.array([1., 1., 1.]), np.array([0., 1., 1.])))
print(tf.losses.mean_pairwise_squared_error(np.array([1., 1., 1.]), np.array([1., 1., 1.])))
#
num = 26
for i in range(num):
    label = np.load('label{}.npy'.format(i))
    output = np.load('output_prob{}.npy'.format(i))
    computed_loss = np.load('computed_loss{}.npy'.format(i))
    print(label[0, :, :, 0].shape)
    plt.subplot(1, 2, 1)
    plt.imshow(output[0, :, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(label[0, :, :, 0])

    print('###################')
    print(tf.losses.mean_pairwise_squared_error(np.array([1., 1., 1.]), np.array([1., 1., 1.])))
    print(tf.losses.mean_pairwise_squared_error(label, label))
    print(tf.losses.mean_pairwise_squared_error(label, output))

    plt.show()

