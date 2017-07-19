# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files for the Behavioral Cloning Project.

In this project, we use what we've learned about deep neural networks and convolutional neural networks to clone driving behavior. We train, validate and test a model using TensorFlow's Keras API. The model outputs a steering angle to an autonomous vehicle.

A simulator is provided where we can steer a car around a track for data collection using a joystick. We use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

The project includes five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* this README report writeup 
* video.mp4 (a video recording of the vehicle driving autonomously around the track for at least one full lap)


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with this written report


Setup Requirements
---
The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)


The following resources can be found in the udacity github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the Udacity classroom along with some sample data.

[//]: # (Image References)

[image1]: ./simulator_image.png "Model Visualization"
[image2]: ./figure_1.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

Using the Simulator to Collect Data
---
Using a joystick and driving a couple of labs around the track we generate left, right and center image data along with also recording the steering angel. This data is recorded using the simulator GUI and saved onto the data subdirectory for processing next.

![alt text][image1]

After the collection process, I had 3,000 number of data points. I augmented this data by flipping vertically, I then randomly shuffled the data set and put 35% of the data into a validation set. 


Creating a Trained Model
---

We create the python model.py script which trains a model to learn the driving behavior recorded previously.

The intial step for the model is to normalize the data.

The second step is to crop out the distraction parts of the images which are mainly the background scenery.

The second step is followed by three 5x5 convolutional layers and two 3x3 convolutional layers.

The convolutional layers gradually increase the depth from 24 to 64 as the convolutional sizes decreases from 5x5 to 3x3.

The model includes RELU layers to introduce nonlinearity.

Lastly, the model flattens and is compiled.


Starting a Training Session
---

No dropout layers to reduce overfitting were implemented. We see the training set fit nicely over 10 iterations whereas the validation sets seems to suffer from overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


![alt text][image2]


The model uses an adam optimizer, so the learning rate was not tuned manually.

Training data was chosen to keep the vehicle driving on the road. We used a combination of center lane driving, recovering from the left and right sides of the road .

At the end of the process, the vehicle is able to drive autonomously around the track with leaving the road.




I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.



