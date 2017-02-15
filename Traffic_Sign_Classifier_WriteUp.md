
##Writeup 

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I just use the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.hist(y_train, n_classes, normed=1, histtype='bar', facecolor='b', alpha=0.75)
ax.set_title('label frequency')
ax.set_xlabel('labels')
ax.set_ylabel('frequency')

plt.show()
```

Here is an exploratory visualization of the data set. Here I use matplot library function `ax.hist()` to directly plot the frequency of the labels.
![这里写图片描述](http://img.blog.csdn.net/20170215192922490?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is list here:
```python
import cv2

# convert color image to gray-scale image，here we use opencv api function, 
# gray= R*0.299+G*0.587+B*0.114
def color2gray(image):
    gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray_img

def Images_color2Gray(image_group):
    X_train_gray=[]
    for train_image in image_group: 
        gray_img=color2gray(train_image)
        gray_img = gray_img[:, :, np.newaxis]
        X_train_gray.append(gray_img)
    
    return X_train_gray
##################################
# plot two images
fig, axs = plt.subplots(1,2, figsize=(5, 2))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('raw')
axs[0].imshow(X_train[4900].squeeze())

# transfer gray images to X_train and X_test
X_train = Images_color2Gray(X_train)
X_test = Images_color2Gray(X_test)

axs[1].axis('off')
axs[1].set_title('gray')
axs[1].imshow(X_train[4900].squeeze(), cmap='gray')
```
As a first step, I decided to convert the images to grayscale because for the RGB images, it has three channel for each pixels, however the gray images only has one channel for each pixels, which can reduce the use of the computational load and storage cost. 

Here is an example of a traffic sign image before and after grayscaling.

![这里写图片描述](http://img.blog.csdn.net/20170215193807314?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

As a last step, I normalized the image data because if the input value size is too big or too small, it will effect the loss function when training the model. If the input value has zero means and equal variance, it will have better training accuracy and results.

There is a [wiki](http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing#Color_Images) explaining three methods to pre-process data.
> - Simple Rescaling:In simple rescaling, our goal is to rescale the data along each data dimension (possibly independently) so that the final data vectors lie in the range [0,1] or [ − 1,1] (depending on your dataset). In MNIST project, we can normalize the pixels from [0, 255] to [0, 1]. 
- Per-example mean subtraction: If your data is stationary (i.e., the statistics for each data dimension follow the same distribution), then you might want to consider subtracting the mean-value for each example (computed per-example).
- Feature Standardization: Feature standardization refers to (independently) setting each dimension of the data to have zero-mean and unit-variance. This is the most common method for normalization and is generally used widely (e.g., when working with SVMs, feature standardization is often recommended as a preprocessing step). In practice, one achieves this by first computing the mean of each dimension (across the dataset) and subtracts this from each dimension. Next, each dimension is divided by its standard deviation.

 Here I use simple Rescaling  method and convert data range from [0,255] to [-0.5,0.5]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained here:
```python
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train_norm, y_train, 
                                                                test_size=0.20, random_state=42)

print("X_train has {} data ".format(len(X_train)))
print("X_validation has {} data ".format(len(X_validation)))
print("X_test has {} data ".format(len(X_test)))
```

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn API function `train_test_split` with 20% split percentage.

My final training set had **27839**  number of images. My validation set and test set had **6960**  and **12630** number of images.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. The final model architecture is same as the model in LeNet project. Besides, I also read the [article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), in this article, a new article is explained. The model structure in this article has a few differences in the flatten layers where there is additional convolutional nn layers parallel to the flatten layer. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Dropout	      		| 0.5 keep probability							|
| Flatten 				| outputs 400        							|
| Fully connected		| outputs 120        							|
| RELU					| 		      	 								|
| Fully connected		| outputs 84        							|
| RELU					| 		      	 								|
| Fully connected		| outputs 43        							|
| Softmax				| etc.        									|

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located here:
```python 
rate = 0.001
EPOCHS=30
BATCH_SIZE=128
save_file = './model.ckpt'

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
###################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:0.5})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, save_file)
    print("Model saved")

```
First, I calculate the loss function with `tf.nn.softmax_cross_entropy_with_logits()`, and use `tf.reduce_mean()` as the loss function. 
Then I use Adam method as the optimizer to reduce the loss and adjust the weights in the model. 
The I run the model to find the final solutions. At beginning, I use 128 as the batch size, 0.001 as the learning rate and 30 as the epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of **0.996**
* validation set accuracy of **0.980**
* test set accuracy of **0.917**

Tuning History:
> - Learning rate=0.001,epochs=30, batch_size=128,sigma=0.1
>     **Validation Accuracy = 0.950**
>    - Learning rate=0.001,epochs=60, batch_size=128,sigma=0.1
>     **Validation Accuracy = 0.963**
>     - Learning rate=0.001,epochs=60, batch_size=250,sigma=0.1
>     **Validation Accuracy = 0.961**
>     - change the model structure, move the dropout layer before the last full connected layer, &&   Learning rate=0.001,epochs=60, batch_size=250,sigma=0.1
>     **Validation Accuracy = 0.979**
>     Learning rate=0.001,epochs=60, batch_size=250,sigma=0.1->0.2
>     **Validation Accuracy = 0.971**
>      Learning rate=0.001,epochs=90, batch_size=250,sigma=0.1
>     **Validation Accuracy = 0.976**
>     Learning rate=0.0005,epochs=60, batch_size=250,sigma=0.1
>     **Validation Accuracy = 0.967**
>     Learning rate=0.001,epochs=60, batch_size=128,sigma=0.1
>     **Validation Accuracy = 0.977**
>     Learning rate=0.001,epochs=80, batch_size=128,sigma=0.1
>     **Validation Accuracy = 0.980**


I use the classical model structure from udacity lessons called LeNet. The LeNet consists of a convolutional layer with 32x32x1 gray image input and 28x28x1 as output, followed by a max-pooling layer and relu activation layer. and then we have same three layers following., the final output is 5x5x16. The a Flatten layer was used to compress the 5x5x16 to 400. and then followed by two full connected layer that convert 400 to number of classes 43. 
I tune the epochs, batch size ,learning rate and hyperparameters sigma and mu. Finally I choose LeNet Structure and following parameters:
> - epochs=80
> - batch size=128
> - learning rate =0.001
> - keep_prob=0.5
> - sigma=0.1
> - mu =0

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![这里写图片描述](http://img.blog.csdn.net/20170215231357393?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170215231431612?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170215231451191?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170215231515582?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170215231534426?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

The pictures I choose all have clear background, I think all of them can be detected correctly with high probabilities.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.
```python
saver = tf.train.Saver()
# create predictor 
predictor = tf.argmax(tf.nn.softmax(logits), 1)

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)
    out_labels=sess.run(predictor, feed_dict={x: m_norm,keep_prob:0.5})

print(out_labels)
   
```
Here are the results of the prediction:

![这里写图片描述](http://img.blog.csdn.net/20170215232336186?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYTg1MzQ1OTAwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield	      		| Yield	  									| 
| No entry     			| No entry     										|
| Road Work				| Road Work											|
| Keep right      		| Keep right					 				|
| Wild animals crossing			| Wild animals crossing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.7%.

All the images was corrected detected by the trained model, however, the pictures that I choose from the web all have clear backgroud and there are no big noise like trees, bushes, building walls. 
I tried to detect the sign picture which located in front of trees, the tree leaves give the picture big disturbs and the picture was detected with wrong results.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
```python
saver = tf.train.Saver()
# create top_k function
output_topk = tf.nn.top_k(tf.nn.softmax(logits), k=5)

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)
    topk_result=sess.run(output_topk, feed_dict={x: m_norm,keep_prob:0.5})
    
print(topk_result)
```
For the first image, the model is relatively sure that this is a Yield sign (probability of 1.0), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Yield   									| 
| 0.0     				| Speed limit (20km/h)										|
| 0.0					| Speed limit (30km/h)											|
| 0.0	      			| Speed limit (50km/h)					 				|
| 0.0				    | Speed limit (60km/h)      							|

The other four images all have high probabilities that close to 1.0.
The output results of top_k softmax probabilities is listed below:
```python
TopKV2(values=array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   6.34437101e-16,   4.04985601e-23,
          5.15150586e-24,   4.19998125e-25],
       [  1.00000000e+00,   4.34199518e-23,   1.69864686e-28,
          4.13321446e-31,   2.39948725e-31],
       [  8.79261076e-01,   1.19257376e-01,   1.39593752e-03,
          8.48669515e-05,   4.63925517e-07],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00]], dtype=float32), indices=array([[13,  0,  1,  2,  3],
       [17, 14, 33,  0, 20],
       [25, 20,  4, 26,  5],
       [38, 31, 23, 18, 29],
       [31,  0,  1,  2,  3]], dtype=int32))
```
