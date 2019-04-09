import model as mod
import tensorflow as tf
import RewardManager as RM
import functools as func
import numpy as np
import matplotlib.pyplot as plt
import divers as div
import cv2
import scipy as sc


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        self.myModel = mod.Reinforcement()
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9)
        self.action = RM.RewardManager()
        self.width, self.height = 0, 0
        self.best_idx, self.future_reward = [0, 0], 0
        self.scale_factor = 2
        self.output_prob = 0
        self.loss_value = 0
        self.iteration = 0

        ###### A changer par une fonction adaptée au demonstration learning ######
        self.loss = func.partial(tf.losses.huber_loss)  # Huber loss
        ######                     Demonstration                            ######
        self.best_idx = [125, 103]

    def forward(self, input):
        self.image = input
        # Increase the size of the image to have a relevant output map
        input = div.preprocess_img(input, target_height=self.scale_factor*224, target_width=self.scale_factor*224)
        # Pass input data through model
        self.output_prob = self.myModel(input)
        # Useless for the moment
        self.width, self.height = self.output_prob.shape[1], self.output_prob.shape[2]
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
        label, label_weights = self.compute_labels(expected_reward, self.best_idx)
        self.output_prob = tf.reshape(self.output_prob, (self.width, self.height, 1))
        self.loss_value = self.loss(label, self.output_prob, label_weights)
        return(self.loss_value)

    def max_primitive_pixel(self, prediction):
        '''Locate the max value-pixel of the image
        Locate the highest pixel of a Q-map
        :param prediction: Q map
        :return: max_primitive_pixel_idx (tuple) : pixel of the highest Q value
                 max_primitive_pixel_value : value of the highest Q-value
        '''
        # Transform the Q map tensor into a 2-size numpy array
        numpy_predictions = prediction.numpy()[0, :, :, 0]
        viz = False
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

        # Best Idx in image frame
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
        label = np.zeros((224, 224, 1))
        area = 7
        label[best_pix_ind[0]-area:best_pix_ind[0]+area, best_pix_ind[1]-area:best_pix_ind[1]+area] = label_value
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        # Compute label mask
        label_weights = np.zeros(label.shape)
        label_weights[best_pix_ind[0]-10:best_pix_ind[0]+10, best_pix_ind[1]-10:best_pix_ind[1]+10] = 1

        # plt.subplot(1, 3, 1)
        # self.image = np.reshape(self.image, (self.image.shape[1], self.image.shape[2], 3))
        # plt.imshow(self.image)
        # plt.subplot(1, 3, 2)
        # label_viz = np.reshape(label, (label.shape[0], label.shape[1]))
        # plt.imshow(label_viz)

        label, label_weights = tf.convert_to_tensor(label, np.float32),\
                               tf.convert_to_tensor(label_weights, np.float32)
        label, label_weights = tf.image.resize_images(label, (self.width, self.height)),\
                               tf.image.resize_images(label_weights, (self.width, self.height))
        # plt.subplot(1, 3, 3)
        # plt.imshow(label.numpy()[:, :, 0])
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


if __name__ == '__main__':
    Network = Trainer()
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[0, 70:190, 100:105, :] = 1
    im[0, 70:80, 80:125, :] = 1
    best_pix = [125, 103]
    test = False
    if test:
        result = Network.forward(im)
        best_idx, best_value, image_idx, image_value = Network.get_best_predicted_primitive()
        Network.vizualisation(im, image_idx)
        result = tf.reshape(result, (result.shape[1], result.shape[2]))

    previous_qmap = Network.forward(im)
    N = 5
    for i in range(N):
        print('Iteration {}/{}'.format(i, N-1))
        Network.main(im)
    trained_qmap = Network.forward(im)

    # Creation of a rotated view
    im2 = sc.ndimage.rotate(im[0, :, :, :], 90)
    im2.reshape(1, im2.shape[0], im2.shape[1], im2.shape[2])
    im2 = np.array([im2])

    # Resizes images
    new_qmap = Network.forward(im2)
    trained_qmap = tf.image.resize_images(trained_qmap, (14, 14))
    previous_qmap = tf.image.resize_images(previous_qmap, (14, 14))
    new_qmap = tf.image.resize_images(new_qmap, (14, 14))
    print(new_qmap.shape)

    # Plotting
    plt.subplot(1, 3, 3)
    plt.imshow(tf.reshape(new_qmap, (14, 14)))
    plt.subplot(1, 3, 1)
    plt.imshow(tf.reshape(previous_qmap, (14, 14)))
    plt.subplot(1, 3, 2)
    plt.imshow(tf.reshape(trained_qmap, (14, 14)))
    plt.show()
