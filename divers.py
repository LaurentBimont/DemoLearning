import tensorflow as tf
import numpy as np


def rotate_image2( input_data, input_angles):
    return tf.contrib.image.rotate(input_data, input_angles, interpolation="BILINEAR")


def preprocess_img(img, target_height=224*5, target_width=224*5, rotate=False):
    # Apply 2x scale to input heightmaps
    resized_img = tf.image.resize_images(tf.Variable(img), (target_height, target_width))

    # Peut être rajouter un padding pour éviter les effets de bords

    if rotate:
        rimgs = rotate_image2(resized_imgs, list_angles)

        # Add extra padding (to handle rotations inside network)
        diag_length = float(target_height) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - target_height))

        padded_imgs = tf.image.resize_image_with_crop_or_pad(rimgs,target_height+padding_width,target_width+padding_width)

        return padded_imgs, padding_width

    return resized_img


def postprocess_img( imgs, list_angles):
    # Return Q values (and remove extra padding
    # Reshape to standard
    resized_imgs = tf.image.resize_images(imgs, (320, 320))
    # Perform rotation
    rimgs = rotate_image2(resized_imgs, list_angles)
    # Reshape rotated images
    resized_imgs = tf.image.resize_images(rimgs, (320, 320))
    return resized_imgs
