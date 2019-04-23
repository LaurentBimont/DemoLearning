import model as mod
import tensorflow as tf
import rewardManager as RM
import functools as func
import numpy as np
import matplotlib.pyplot as plt
import divers as div
#import cv2
import scipy as sc
import dataAugmentation as da
from threading import Thread, Lock, Barrier

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        self.myModel = mod.Reinforcement()
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=5e-4, momentum=0.9)
        self.action = RM.RewardManager()
        self.width, self.height = 0, 0
        self.best_idx, self.future_reward = [0, 0], 0
        self.scale_factor = 1
        self.output_prob = 0
        self.loss_value = 0
        self.iteration = 0

        ###### A changer par une fonction adaptée au demonstration learning ######
        self.loss = func.partial(tf.losses.huber_loss)  # Huber loss
        ######                     Demonstration                            ######
        self.best_idx = [125, 103]

    def custom_loss(self):
        '''
        As presented in 'Deep Q-learning from Demonstrations, the loss value is highly impacted by the
        :return: Loss Value
        '''
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * 0.001

    def forward(self, input):
        self.image = input
        # Increase the size of the image to have a relevant output map
        input = div.preprocess_img(input, target_height=self.scale_factor*224, target_width=self.scale_factor*224)
        # Pass input data through model
        self.output_prob = self.myModel(input)
        # print(self.output_prob.numpy()[0, :, :, :].shape, type(self.output_prob.numpy()[0, :, :, :]))
        # plt.imshow(self.output_prob.numpy()[0, :, :, 0])
        # plt.show()
        # Useless for the moment
        self.batch, self.width, self.height = self.output_prob.shape[0], self.output_prob.shape[1], self.output_prob.shape[2]
        # Return Q-map
        return self.output_prob

    def backpropagation(self, gradient):
        self.optimizer.apply_gradients(zip(gradient, self.myModel.trainable_variables),
                                       global_step=None)        # tf.train.get_or_create_global_step())
        self.iteration = tf.train.get_global_step()

    def compute_loss(self):
        # A changer pour pouvoir un mode démonstration et un mode renforcement
        expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
        label224, label_weights224 = self.compute_labels(expected_reward, self.best_idx)

        label, label_weights = self.reduced_label(label224, label_weights224)
        self.output_prob = tf.reshape(self.output_prob, (self.width, self.height, 1))
        self.loss_value = self.loss(label, self.output_prob, label_weights)
        return self.loss_value

    def compute_loss_dem(self, label, label_w, viz=False):
        expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
        if viz:
            plt.imshow(label[0, :, :, :])
            plt.show()
        label, label_weights = self.reduced_label(label, label_w)
        self.output_prob = tf.reshape(self.output_prob, (self.batch, self.width, self.height, 1))
        self.loss_value = self.loss(label, self.output_prob, label_weights)
        return self.loss_value

    def max_primitive_pixel(self, prediction, viz=False):
        '''Locate the max value-pixel of the image
        Locate the highest pixel of a Q-map
        :param prediction: Q map
        :return: max_primitive_pixel_idx (tuple) : pixel of the highest Q value
                 max_primitive_pixel_value : value of the highest Q-value
        '''
        # Transform the Q map tensor into a 2-size numpy array
        numpy_predictions = prediction.eval()[0, :, :, 0]
        if viz:
            result = tf.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
            plt.subplot(1, 2, 1)
            plt.imshow(result)
            plt.subplot(1, 2, 2)
            plt.imshow(numpy_predictions)
            plt.show()
        # Get the highest pixel
        max_primitive_pixel_idx = np.unravel_index(np.argmax(numpy_predictions),
                                                   numpy_predictions.shape)
        # Get the highest score
        max_primitive_pixel_value = numpy_predictions[max_primitive_pixel_idx]
        print('Grasping confidence scores: {}, {}'.format(max_primitive_pixel_value, max_primitive_pixel_idx))
        return max_primitive_pixel_idx, max_primitive_pixel_value

    def get_best_predicted_primitive(self):
        '''
        :param output_prob: Q-map
        :return: best_idx (tuple): best idx in raw-Q-map
                 best_value : highest value in raw Q-map
                 image_idx (tuple): best pixels in image format (224x224) Q-map
                 image_value : best value in image format (224x224) Q-map
        '''

        # Best Idx in image frameadients(
        prediction = tf.image.resize_images(self.output_prob, (224, 224))
        image_idx, image_value = self.max_primitive_pixel(prediction)
        # Best Idx in network output frame
        best_idx, best_value = self.max_primitive_pixel(self.output_prob)

        self.best_idx, self.future_reward = best_idx, best_value
        return best_idx, best_value, image_idx, image_value

    def compute_labels(self, label_value, best_pix_ind, viz=False):
        '''Create the targeted Q-map
        :param label_value: Reward of the action
        :param best_pix_ind: Pixel where to perform the action
        :return: label : an 224x224 array where best pix is at future reward value
                 label_weights : a 224x224 where best pix is at one
        '''
        # Compute labels
        label = np.zeros((224, 224, 3), dtype=np.float32)
        area = 7
        label[best_pix_ind[0]-area:best_pix_ind[0]+area, best_pix_ind[1]-area:best_pix_ind[1]+area, :] = label_value
        label_weights = np.zeros(label.shape, dtype=np.float32)
        label_weights[best_pix_ind[0]-area:best_pix_ind[0]+area, best_pix_ind[1]-area:best_pix_ind[1]+area, :] = 1

        if viz:
            plt.subplot(1, 3, 1)
            self.image = np.reshape(self.image, (self.image.shape[1], self.image.shape[2], 3))
            plt.imshow(self.image)
            plt.subplot(1, 3, 2)
            label_viz = np.reshape(label, (label.shape[0], label.shape[1]))
            plt.imshow(label_viz)
        return label, label_weights

    def reduced_label(self, label, label_weights, viz=False):
        '''Reduce label Q-map to the output dimension of the network
        :param label: 224x224 label map
        :param label_weights:  224x224 label weights map
        :return: label and label_weights in output format
        '''

        if viz:
            plt.subplot(1, 2, 1)
            plt.imshow(label[0, :, :, 0])

        label, label_weights = tf.convert_to_tensor(label, np.float32),\
                               tf.convert_to_tensor(label_weights, np.float32)
        print('la taille des label ', label.shape)

        label, label_weights = tf.image.resize_images(label, (self.width, self.height)),\
                               tf.image.resize_images(label_weights, (self.width, self.height))
        print('la taille des label ', label.shape)
        print('la taille à forcer ', self.batch, self.width, self.height, 1)
        label, label_weights = tf.reshape(label[:, :, :, 0], (self.batch, self.width, self.height, 1)), \
                               tf.reshape(label_weights[:, :, :, 0], (self.batch, self.width, self.height, 1))
        if viz:
            plt.subplot(1, 2, 2)
            plt.imshow(label.eval()[0, :, :, 0])
            plt.show()

        return label, label_weights

    def main(self, input):
        self.future_reward = 1
        print(type(input))
        self.forward(input)
        self.compute_loss_dem()

        train_op = self.optimizer.minimize(loss=self.loss_value, global_step=tf.train.get_global_step())

        grad = self.optimizer.compute_gradients(self.loss_value)
        self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())


    def main_augmentation(self, dataset):
        ima, val, val_w = dataset['im'], dataset['label'], dataset['label_weights']
        self.future_reward = 1
        for j in range(len(ima)):
            if j % 10 == 0:
                print('Iteration {}/{}'.format(j, len(ima)))
            with tf.GradientTape() as tape:
                self.forward(tf.reshape(ima[j], (1, 224, 224, 3)))
                self.compute_loss_dem(val[j], val_w[j])
                grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                               global_step=tf.train.get_or_create_global_step())

    def main_batches(self, im, label, label_weights, viz=False):
        self.future_reward = 1
        print(im.shape)
        self.forward(im)
        if viz:
            plt.imshow(label[0])
            plt.show()
        self.compute_loss_dem(label, label_weights, viz=False)
        train_op =self.optimizer.apply_gradients(zip(grad, self.myModel.my_trainable_variables),
                                       global_step=tf.train.get_or_create_global_step())
        train_op = self.optimizer.minimize(self.loss_value, global_step=tf.train.get_or_create_global_step())
        tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=self.loss_value, train_op=train_op)


if __name__ == '__main__':

    Network = Trainer()
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    best_pix = [125, 103]

    Network.forward(im)
    batch_size = 3

    label, label_weights = Network.compute_labels(1.8, best_pix)
    dataset = da.OnlineAugmentation().generate_batch(im, label, label_weights, augmentation_factor=2, viz=False)
    im_o, label_o, label_wo = dataset['im'], dataset['label'], dataset['label_weights']
    batch_tmp_im, batch_tmp_lab, batch_tmp_weights = [], [], []

    for i in range(batch_size):
        ind_tmp = np.random.randint(len(dataset['im']))
        batch_tmp_im.append(im_o[ind_tmp])
        batch_tmp_lab.append(label_o[ind_tmp])
        batch_tmp_weights.append(label_wo[ind_tmp])

    batch_im, batch_lab, batch_weights = tf.stack(batch_tmp_im), tf.stack(batch_tmp_lab), tf.stack(
        batch_tmp_weights)

    Network.main_batches(batch_im, batch_lab, batch_weights)

    # Test de Tensorboard
    test6 = False
    if test6:
        previous_qmap = Network.forward(im)
        label, label_weights = Network.compute_labels(1.8, best_pix)
        dataset = da.OnlineAugmentation().generate_batch(im, label, label_weights, viz=False)
        im_o, label_o, label_wo = dataset['im'], dataset['label'], dataset['label_weights']
        epoch_size = 1
        batch_size = 2
        for epoch in range(epoch_size):
            for batch in range(len(dataset['im']) // batch_size):
                print('Epoch {}/{}, Batch {}/{}'.format(epoch + 1, epoch_size, batch + 1,
                                                        len(dataset['im']) // batch_size))
                batch_tmp_im, batch_tmp_lab, batch_tmp_weights = [], [], []
                for i in range(batch_size):
                    ind_tmp = np.random.randint(len(dataset['im']))
                    batch_tmp_im.append(im_o[ind_tmp])
                    batch_tmp_lab.append(label_o[ind_tmp])
                    batch_tmp_weights.append(label_wo[ind_tmp])

                batch_im, batch_lab, batch_weights = tf.stack(batch_tmp_im), tf.stack(batch_tmp_lab), tf.stack(
                    batch_tmp_weights)

                Network.main_batches(batch_im, batch_lab, batch_weights)

        trained_qmap = Network.forward(im)

        ntrained_qmap = trained_qmap.eval()

        print(np.argmax(ntrained_qmap), np.argmax(ntrained_qmap[0]))

