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


Creating a Trained Model
---

We create the python model.py script which trains a model to learn the driving behavior recorded previously.

The intial step for the model is to normalize the data.

The second step is to crop out the distraction parts of the images which are mainly the background scenery.

The second step is followed by three 5x5 convolutional layers and two 3x3 convolutional layers.

Lastly, the model flattens and is compiled.

The convolutional layers gradually increase the depth from 24 to 64 as the convolutional sizes decreases from 5x5 to 3x3.

The model includes RELU layers to introduce nonlinearity.


Starting a Training Session
---



The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.



