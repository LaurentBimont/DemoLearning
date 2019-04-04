import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()


class DensenetFeatModel(tf.keras.Model):
    def __init__(self):
        super(DensenetFeatModel, self).__init__()
        baseModel = tf.keras.applications.densenet.DenseNet121(weights='imagenet')
        self.model = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.get_layer(
            "conv5_block16_concat").output)

    def call(self, inputs):
        # inputs = tf.transpose(inputs,(0,3,2,1))
        output = self.model(inputs)
        print(output.shape)
        return output


class BaseDeepModel(tf.keras.Model):
    def __init__(self):
        super(BaseDeepModel, self).__init__()
        pass


class GraspNet(BaseDeepModel):
    def __init__(self):
        super(GraspNet, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167
        self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")
        self.conv0 = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, activation=tf.nn.relu,
                                            use_bias=False, name="grasp-conv0")
        self.bn1 = tf.keras.layers.BatchNormalization(name="grasp-b1")
        self.conv1 = tf.keras.layers.Conv2D(3, kernel_size=1, strides=1, activation=tf.nn.relu, use_bias=False, name="grasp-conv1")
        self.bn2 = tf.keras.layers.BatchNormalization(name="grasp-b2")

    def call(self, inputs, bufferize=False, step_id=-1):
        x = self.bn0(inputs)
        x = self.conv0(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = (x[:, :, :, 0]+x[:, :, :, 1]+x[:, :, :, 2])/3.
        x = tf.reshape(x, (*x.shape, 1))
        x = self.bn2(x)
        return x


class Reinforcement(tf.keras.Model):
    def __init__(self):
        super(Reinforcement, self).__init__()
        self.Dense = DensenetFeatModel()
        self.QGrasp = GraspNet()

    def call(self, input):
        return self.QGrasp(self.Dense(input))


if __name__=="__main__":
    # Rq DenseNet ne semble pas  vouloir seulement du 224*224
    im = np.ones((1, 1280, 800, 3), np.float32)
    print(im.shape)
    model = DensenetFeatModel()
    print('Sortie de mon réseau convolutif', model(im).shape)
    graspModel = GraspNet()
    result = graspModel(model(im)).numpy()
    print(result.shape)
    result = result.reshape((result.shape[1], result.shape[2]))
    plt.imshow(result)
    plt.show()

    rein = Reinforcement()
    result = rein(im).numpy()
    print(result.shape)
    result = result.reshape((result.shape[1], result.shape[2]))
    plt.imshow(result)
    plt.show()
