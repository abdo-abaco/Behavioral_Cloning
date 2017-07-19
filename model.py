import csv
import cv2
import numpy as np


lines = []

# storing camera images and associated steering angles into a log 
with open('data/driving_log.csv') as csvfile:

  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []


# individually processing left, right and center images
for line in lines:
  for i in range(3):
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# augmenting generated images for greater generalization
augmented_images, augmented_measurements = [], []
for image,measurment in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image,1))
  augmented_measurements.append(measurement*-1.0)



#train_generator = imageDataGenerator()


# storing images and steering data onto X_trian and y_train respectively
X_train = np.array(images)
y_train = np.array(measurements)



# Using keras tensorflow model to train network
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

from keras.layers.pooling import MaxPooling2D

# plotting accuracy measures on the convolution neural network
import matplotlib.pyplot as plt


# implementing the layer archetectures
model = Sequential()
#model.add(Reshape((64,128,3), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.35, shuffle=True, nb_epoch=10)

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


model.save('model.h5')










  
