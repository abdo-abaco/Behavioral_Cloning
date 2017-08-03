import csv
import cv2
import numpy as np

import tensorflow as tf
import random
import csv
import cv2 
import json
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda, Reshape, AveragePooling2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2

# plotting accuracy measures on the convolution neural network
import matplotlib.pyplot as plt

def get_csv_data(log_file):
    """
    Reads a csv file and returns two lists separated into examples and labels.
    :param log_file: The path of the log file to be read.
    """
    image_names, steering_angles = [], []
    # Steering offset used for left and right images
    steering_offset = 0.275
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+steering_offset, angle-steering_offset])

    return image_names, steering_angles


def generate_batch(X_train, y_train, batch_size=64):

    images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    while True:
        straight_count = 0
        for i in range(batch_size):
            # Select a random index to use for data sample
            sample_index = random.randrange(len(X_train))
            image_index = random.randrange(len(X_train[0]))
            angle = y_train[sample_index][image_index]
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            # Read image in from directory, process, and convert to numpy array
            image = cv2.imread('data/' + str(X_train[sample_index][image_index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.float32)
            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles







# Get the training data from log file, shuffle, and split into train/validation datasets
X_train, y_train = get_csv_data('data/driving_log.csv')
X_train, y_train = shuffle(X_train, y_train, random_state=14)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=14)

# Get model, print summary, and train using a generator
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(.1))
model.add(Dense(100, activation="relu"))
model.add(Dropout(.3))
model.add(Dense(50, activation="relu"))
model.add(Dropout(.5))
model.add(Dense(10, activation="relu"))
model.add(Dropout(.7))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


history_object = model.fit_generator(generate_batch(X_train, y_train), samples_per_epoch=16000, nb_epoch=12, validation_data=generate_batch(X_validation, y_validation), nb_val_samples=1024)


### print the keys contained in the history object 
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss') 
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print('Saving model weights and configuration file.')
# Save model weights
model.save('model.h5')









  
