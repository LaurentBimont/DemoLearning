import model as mod
import tensorflow as tf
import RewardManager as RM
import functools as func
import numpy as np
import matplotlib.pyplot as plt
import divers as div
import cv2


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        self.myModel = mod.Reinforcement()
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
        self.action = RM.RewardManager()
        self.width, self.height = 0, 0
        self.best_idx, self.future_reward = [0, 0], 0
        self.scale_factor = 5
        self.output_prob = 0
        self.loss_value = 0

        ###### A changer par une fonction adaptée au demonstration learning ######
        self.loss = func.partial(tf.losses.huber_loss)  # Huber loss
        ######                     Demonstration                            ######
        self.best_pix = [125, 103]

    def forward(self, input):
        # Increase the size of the image to have a relevant output map
        input = div.preprocess_img(input, target_height=self.scale_factor*224, target_width=self.scale_factor*224)
        # Pass input data through model
        self.output_prob = self.myModel(input)
        # Useless for the moment
        self.width, self.height = self.output_prob.shape[1], self.output_prob.shape[2]
        # Return Q-map
        return self.output_prob

    def backprop(self, gradient):
        self.optimizer.apply_gradients(zip(gradient, self.myModel.trainable_variables),
        global_step=tf.train.get_or_create_global_step())
        self.iteration = tf.train.get_global_step()

    def compute_loss(self):
        # A changer pour pouvoir un mode démonstration et un mode renforcement
        expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
        label, label_weights = self.compute_labels(expected_reward, self.best_pix)
        self.loss_value = self.loss(self.output_prob, label, label_weights)

    def max_primitive_pixel(self, prediction):
        '''
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
        # Compute labels
        label = np.zeros((224, 224))
        print(label, label.shape)
        print(best_pix_ind[0], best_pix_ind[1])
        print(label_value)
        label[best_pix_ind[0], best_pix_ind[1]] = label_value
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        # Compute label mask
        label_weights = np.zeros(label.shape)
        label_weights[best_pix_ind[0], best_pix_ind[1]] = 1
        return label, label_weights

    def vizualisation(self, img, idx):
        prediction = cv2.circle(img[0], (int(idx[1]), int(idx[0])), 7, (255, 255, 255), 2)
        plt.imshow(prediction)
        plt.show()

    def main(self, input):
        self.forward(input)
        self.compute_loss()
        with tf.GradientTape() as tape:
            self.backprop(tape.gradient(self.compute_loss(), self.myModel.trainable_variables))


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

    Network.main(im)
    #with tf.GradientTape() as tape:
    #    loss =
    #    test.backprop(tape.gradient(loss, test.trainable_variables))


######### remettre les labels en format de sortie du réseau ##########