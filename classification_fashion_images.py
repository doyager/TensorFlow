

###########  Loading Libraries ###########
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#out : 1.9.0

########## Loading Data ##############

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Defining Classes and Lables

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
							 
	
# Exloring the data

# train size is 60000 images, which are 28*28 pixels:
train_images.shape      
#op : (60000, 28, 28)

len(train_lables)
#op: 60000

train_labels
# array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)


# test size is 10000 images, which are 28*28 pixels:

test_images.shape
#op: (10000, 28, 28)

len(test_labels)
#op: 10000


######## Preprocess data #########


# plot the first picture in the train data set
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(True)



# Pixels fall in the range of 0 to 255 , so we scale them to a range of 0 to 1, so used them in a NN model.

# cast datatype of image from integer to float and dividing by 255 
train_images = train_images / 255.0
test_images = test_images / 255.0


# Displaying first 25 images from training, and verify to classes to check readiness to build model

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# model

# setting layers for NN

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels),
to a 1d-array of 28 * 28 = 784 pixels. Think of this layer as unstacking rows of pixels in the image and lining them up. This layer
has no parameters to learn; it only reformats the data.

After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely-connected,
or fully-connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer is a 10-node softmax
layer—this returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the
current image belongs to one of the 10 digit classes.
"""

#compile

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
							


# Train

model.fit(train_images, train_labels, epochs=5)



# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# Predict
predictions = model.predict(test_images)
predictions[0] 
# this prediction would give out a array of 10 number , that is confidence towards each label/class, so which ever is highest value it
is understood that the image belongs to that class

#max confidence value for first test prediciton record
np.argmax(predictions[0])
#op: 9

# comparision
# actual test label data set first record

test_labels[0]
# op : 9



 Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
 
 
 
#usually tf.keras are used to make predicitons on a batch or collection , so for doing it on one item/image 

################## single image ######################
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)
#op: (1, 28, 28)

#predict on single image
predictions = model.predict(img)

print(predictions)
#op: [[1.7615672e-05 5.7607571e-07 2.0116224e-06 3.3666095e-07 6.0833884e-07
#  7.1541648e-03 6.1689807e-06 1.5703385e-01 4.6337722e-04 8.3532131e-01]]
	
#find max value of all in the array
prediction = predictions[0]
np.argmax(prediction)
#op: 9
