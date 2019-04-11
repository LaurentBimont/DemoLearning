import model as mod
import tensorflow as tf
import rewardManager as RM
import functools as func
import numpy as np
import matplotlib.pyplot as plt
import divers as div
#import cv2
import scipy as sc
#import dataAugmentation as da


class Trainer(object):
    def __init__(self):
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # session = tf.Session(config=config)
        # tf.enable_eager_execution()

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
        print(10)
        self.image = input
        # Increase the size of the image to have a relevant output map
        print(11)
        input = div.preprocess_img(input, target_height=self.scale_factor*224, target_width=self.scale_factor*224)
        # Pass input data through model
        print(12)
        print(input.shape, type(input))
        self.output_prob = self.myModel(input)
        # Useless for the moment
        print(13)
        self.batch, self.width, self.height = self.output_prob.shape[0], self.output_prob.shape[1], self.output_prob.shape[2]
        # Return Q-map
        return self.output_prob

    def backpropagation(self, gradient):
        print(np.array(self.myModel.trainable_variables).shape, np.array(gradient).shape)
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
        numpy_predictions = prediction.numpy()[0, :, :, 0]
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

    def compute_labels(self, label_value, best_pix_ind):
        '''Create the targeted Q-map
        :param label_value: Reward of the action
        :param best_pix_ind: Pixel where to perform the action
        :return: label : an 224x224 array where best pix is at future reward value
                 label_weights : a 224x224 where best pix is at one
        '''
        # Compute labels
        label = np.zeros((224, 224, 3))
        area = 7
        label[best_pix_ind[0]-area:best_pix_ind[0]+area, best_pix_ind[1]-area:best_pix_ind[1]+area, :] = label_value
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        # Compute label mask
        label_weights = np.zeros(label.shape)
        label_weights[best_pix_ind[0]-area:best_pix_ind[0]+area, best_pix_ind[1]-area:best_pix_ind[1]+area, :] = 1

        viz = False
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
        # label, label_weights = label[:, :, 0], label_weights[:, :, 0]
        if viz:
            plt.subplot(1, 2, 1)
            plt.imshow(label[0, :, :, 0])

        label, label_weights = tf.convert_to_tensor(label, np.float32),\
                               tf.convert_to_tensor(label_weights, np.float32)
        label, label_weights = tf.image.resize_images(label, (self.width, self.height)),\
                               tf.image.resize_images(label_weights, (self.width, self.height))
        print(label.shape)
        label, label_weights = tf.reshape(label[:, :, :, 0], (self.batch, self.width, self.height, 1)), \
                               tf.reshape(label_weights[:, :, :, 0], (self.batch, self.width, self.height, 1))
        print(label.numpy()[0, :, :, 0].shape)
        if viz:
            plt.subplot(1, 2, 2)
            plt.imshow(label.numpy()[0, :, :, 0])
            plt.show()
        return label, label_weights

    def vizualisation(self, img, idx):
        prediction = cv2.circle(img[0], (int(idx[1]), int(idx[0])), 7, (255, 255, 255), 2)
        plt.imshow(prediction)
        plt.show()

    def main(self, input):
        self.future_reward = 1
        with tf.GradientTape() as tape:
            self.forward(input)
            self.compute_loss()
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
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

    def main_batches(self, im, label, label_weights):
        self.future_reward = 1
        with tf.GradientTape() as tape:
            self.forward(im)
            plt.imshow(label[0])
            plt.show()
            self.compute_loss_dem(label, label_weights, viz=False)
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())


if __name__ == '__main__':

    Network = Trainer()
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    best_pix = [125, 103]

#### Test Initial
    test1 = False
    if test1:
        im = np.zeros((5, 224, 224, 3), np.float32)
        im[:, 70:190, 100:105, :] = 1
        im[:, 70:80, 80:125, :] = 1
        result = Network.forward(im)
        best_idx, best_value, image_idx, image_value = Network.get_best_predicted_primitive()
        Network.vizualisation(im, image_idx)
        result = tf.reshape(result, (result.shape[1], result.shape[2]))

#### Test avec une seule image
    test2 = False
    if test2:
        im = np.zeros((10, 224, 224, 3), np.float32)
        im[:, 70:190, 100:105, :] = 1
        im[:, 70:80, 80:125, :] = 1
        previous_qmap = Network.forward(im)
        label, label_weights = Network.compute_labels(1.8, best_pix)

        print('Stacking in progress')
        label, label_weights = tf.stack([label for i in range(10)]), tf.stack([label_weights for i in range(10)])

        N = 10

        for i in range(N):
            print('Iteration {}/{}'.format(i, N-1))
            Network.main_batches(im, label, label_weights)
        trained_qmap = Network.forward(im)

        # Creation of a rotated view
        im2 = sc.ndimage.rotate(im[0, :, :, :], 90)
        im2.reshape(1, im2.shape[0], im2.shape[1], im2.shape[2])
        im2 = np.array([im2])

        # Resizes images
        new_qmap = Network.forward(im2)
        trained_qmap = tf.image.resize_images(trained_qmap[0], (14, 14))
        previous_qmap = tf.image.resize_images(previous_qmap[0], (14, 14))
        new_qmap = tf.image.resize_images(new_qmap[0], (14, 14))
        print(new_qmap.shape)

        # Plotting
        plt.subplot(1, 3, 3)
        plt.imshow(tf.reshape(new_qmap, (14, 14)))
        plt.subplot(1, 3, 1)
        plt.imshow(tf.reshape(previous_qmap, (14, 14)))
        plt.subplot(1, 3, 2)
        plt.imshow(tf.reshape(trained_qmap, (14, 14)))
        plt.show()

#### Test avec Data Augmentation
    test3 = False
    if test3:
        previous_qmap = Network.forward(im)

        label, label_weights = Network.compute_labels(1.8, best_pix)
        dataset = da.OnlineAugmentation().generate_batch(im, label, label_weights, viz=True)
        Network.main_augmentation(dataset)

        trained_qmap = Network.forward(im)

        # Creation of a rotated view
        im2 = sc.ndimage.rotate(im[0, :, :, :], 90)
        im2.reshape(1, im2.shape[0], im2.shape[1], im2.shape[2])
        im2 = np.array([im2])

        # Resizes images
        new_qmap = Network.forward(im2)
        trained_qmap = tf.image.resize_images(trained_qmap, (Network.width, Network.height))
        previous_qmap = tf.image.resize_images(previous_qmap, (Network.width, Network.height))
        new_qmap = tf.image.resize_images(new_qmap, (Network.width, Network.height))
        print(new_qmap.shape)

        # Plotting
        plt.subplot(1, 3, 3)
        plt.imshow(tf.reshape(new_qmap, (Network.width, Network.height)))
        plt.subplot(1, 3, 1)
        plt.imshow(tf.reshape(previous_qmap, (Network.width, Network.height)))
        plt.subplot(1, 3, 2)
        plt.imshow(tf.reshape(trained_qmap, (Network.width, Network.height)))
        plt.show()

#### Test avec Data Augmentation et Batch
    test4 = True
    if test4:
        previous_qmap = Network.forward(im)
        label, label_weights = Network.compute_labels(1.8, best_pix)

        dataset = da.OnlineAugmentation().generate_batch(im, label, label_weights, viz=False)

        im_o, label_o, label_wo = dataset['im'], dataset['label'], dataset['label_weights']

        plt.subplot(1, 3, 1)
        plt.imshow(im_o[0])
        plt.subplot(1, 3, 2)
        plt.imshow(label_o[0])
        plt.subplot(1, 3, 3)
        plt.imshow(label_wo[0])
        plt.show()

        epoch_size = 1
        batch_size = 8
        for epoch in range(epoch_size):

            for batch in range(len(dataset['im'])//batch_size):

                print('Epoch {}/{}, Batch {}/{}'.format(epoch+1, epoch_size, batch+1, len(dataset['im'])//batch_size))
                batch_tmp_im, batch_tmp_lab, batch_tmp_weights = [], [], []
                for i in range(10):
                    ind_tmp = np.random.randint(len(dataset['im']))
                    batch_tmp_im.append(im_o[ind_tmp])
                    batch_tmp_lab.append(label_o[ind_tmp])
                    batch_tmp_weights.append(label_wo[ind_tmp])

                batch_im, batch_lab, batch_weights = tf.stack(batch_tmp_im), tf.stack(batch_tmp_lab), tf.stack(batch_tmp_weights)

                Network.main_batches(batch_im, batch_lab, batch_weights)

        trained_qmap = Network.forward(im)

        # Creation of a rotated view
        im2 = sc.ndimage.rotate(im[0, :, :, :], 90)
        im2.reshape(1, im2.shape[0], im2.shape[1], im2.shape[2])
        im2 = np.array([im2])

        # Resizes images
        new_qmap = Network.forward(im2)
        trained_qmap = tf.image.resize_images(trained_qmap, (Network.width, Network.height))
        previous_qmap = tf.image.resize_images(previous_qmap, (Network.width, Network.height))
        new_qmap = tf.image.resize_images(new_qmap, (Network.width, Network.height))
        print(new_qmap.shape)

        # Plotting
        plt.subplot(2, 3, 3)
        # plt.imshow(tf.reshape(new_qmap, (Network.width, Network.height)))
        new_qmap = tf.image.resize_images(new_qmap, (224, 224))
        plt.imshow(tf.reshape(new_qmap, (224, 224)))

        plt.subplot(2, 3, 1)
        # plt.imshow(tf.reshape(previous_qmap, (Network.width, Network.height)))
        previous_qmap = tf.image.resize_images(previous_qmap, (224, 224))
        plt.imshow(tf.reshape(previous_qmap, (224, 224)))

        plt.subplot(2, 3, 2)
        # plt.imshow(tf.reshape(trained_qmap, (Network.width, Network.height)))
        trained_qmap = tf.image.resize_images(trained_qmap, (224, 224))
        plt.imshow(tf.reshape(trained_qmap, (224, 224)))

        plt.subplot(2, 3, 5)
        plt.imshow(im[0, :, :, :])

        plt.subplot(2, 3, 6)
        plt.imshow(im2[0, :, :, :])

        plt.show()
