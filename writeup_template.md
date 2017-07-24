**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Writeup_Images/labels_distribution.png "Visualization"
[image2]: ./Writeup_Images/before_preprocessing.png "Before preprocessing"
[image3]: ./Writeup_Images/after_preprocessing.png "After preprocessing"
[image4]: ./Writeup_Images/1.png "Traffic Sign 1"
[image5]: ./Writeup_Images/2.png "Traffic Sign 2"
[image6]: ./Writeup_Images/3.png "Traffic Sign 3"
[image7]: ./Writeup_Images/4.png "Traffic Sign 4"
[image9]: ./Writeup_Images/5.png "Traffic Sign 5"
[image10]: ./Writeup_Images/softmax_probabilities.png "Softmax probabilities"
[image11]: ./Writeup_Images/featuremaps.png "Feature Maps"
[image12]: ./Writeup_Images/test_after_preprocessing.png.png "Test Image"

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the classes

![alt text][image1]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because I think that the shape of the sign is what determines its class not it's color 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data so that it shares a similiar distribution, this helps learning  by making smoother (less oscillations) and a bit faster.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 				|
| Fully connected		| Input 2304, output 1152	|
| RELU					|												|
| Dropout					|	probability of 0.5		|
| Fully connected		| Input 1152, output 576	|
| RELU					|												|
| Dropout					|	probability of 0.5		|
| Fully connected		| Input 576, output 43	|
| Softmax				|    									|
 

To train the model, I used a learning rate 0.001, it is widely considered a good starting point in many cases and since I'm using Adam as an optimizer, which is an adaptive optimization method, the learning rate was good enough to reach good performance.
I have used a relatively large batch size, 256, since the images are only 32x32 grayscale images, that is to have more accurate gradient updates, and trained the model for 30 epochs, which was enough to reach a good performance, going further resulted in decreased validation accuracy.


My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.931
* test set accuracy of 0.918

For the testing set extra metrics were calculated:

* Recall = 0.918
* Precision = 0.920

I have used LeNet architecture as it's a simple yet powerful and proven architecture, and it was already used for similiar sized images,I slightly modofied it by:

Increasing the filters in the first layers to 32
Increasing the filters in the first layers to 64
Increasing the first fully connected layer size to 1152 which is half the output from the previous flattened layer
Increasing the second fully connected layer size to 576 which is half the output from the previous fully connected layer
Adding drop-out layers after each fully connected layer with 0.5 probability

Although the accuracy on training, validating, testing data shows slight overfitting, it also indicates that the network can effectively learn important features from the images to reach acceptable accuracy on unseen data 93.1% on validation and 91.8% on test set

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image9]

The first image might be difficult to classify because it is very similar to the second one except for the reversed part (pointing right or left)

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Dangerous curve to the left      		| Dangerous curve to the left   									| 
| Dangerous curve to the right     			| Dangerous curve to the right										|
| Double curve					| Double curve											|
| Slippery roa	      		| Slippery road				 				|
| Pedestrians			| Pedestrians      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image9]

Here is the activation for the first conv layer for a test image

![alt text][image11]


