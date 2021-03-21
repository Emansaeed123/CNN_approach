#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eman-saeed
"""

#CNN MODEL 
#Part -1 -Building the CNN
"""
- A convolutional neural network is just a artificial neural network , 
  in which you use this convolution trick to add some convolutional layers.
- we make the convolution trick to preserve the special structure in images and classify some images.
- we are going to classify the image for a specific sign for a sign languages.

"""

#Importing the Keras libraries and packages 
"""
- First package is sequential used to intialize our neural network , 
  as there are two ways of initializing a neural network : 
      1- sequence of layers .
      2- graph.
- Second package is Convolution2D is the package that we'll use for the first step of making the CNN,
  which is we add the convolutional layeres .
  the Convolution2D package deal with the images.
  
- the third package  : MaxPooling2D that used in add our pooling layers .
- the fourth package : Flatten that used for convert the pooled feature maps ,
                       that we created through convolution and max pooling into this large feature vector ,
                       and it will be input of our fully connectedd layers.
-the last package : Dense is a package we use to add the fully connected layers and a classic ANN .


"""
#Intialising the CNN
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
tf.__version__

# Importing the dataset
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# Splitting the dataset to get the categorical variable
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Reshaping the independent values into 28x28 pixel groups
X_train = np.array([np.reshape(i, (28,28)) for i in X_train])
X_test = np.array([np.reshape(i, (28,28)) for i in X_test])

# Reshaping the dependant variable
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)

# Encoding categorical data
y_train = tf.keras.utils.to_categorical(y_train, 26)
y_test = tf.keras.utils.to_categorical(y_test, 26)

X_train = X_train.reshape((27455, 28, 28,1))
X_test = X_test.reshape((7172, 28, 28,1))

# Building the CNN

# Initialising the CNN
classifier = tf.keras.models.Sequential()
#Step 1 -Convolution =>
"""
- in this step we are applying a several feature detectors on the input images.
- after applying the input image with the feature detector (filter) we find the Feature Map.
- a feature map contains sum numbers and the highest numbers of the feature map is where in the feature detector could detect a specific feature in the input image.
- the result of the feature map is the convolution operation .
"""
"""
- we use add() method to create a convolutional layer .
- add method has multiple parameters: 
    1-Convolution2D () has miltple arguments : 
        1- nb_filter which is number of filter ->32,
           then we composed of 32 feature maps.
        2- the dimensions of the feature detector (filter) 3 x 3 .
        3- input shape : the shape of your input image ,
           not all the images in the same format so we have to force 
           all images in the same format,so we will make all image in the same format 
           and the same fixed size, it will be in another part of code,
           but in this argument we enter the expected format of our format input images,
           is that B/W image or Colored image.
       4- the activation function :  we use the relu activation function .
"""


# Step 1 - Convolution
classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 1]))

#Step 2 - Pooling 
"""
- we applying the pooling layer becouse we want to reduce the number of nodes we'll get in the next step.
- we use the MaxPooling2D class with some parameters: 
    1- pool_size the most default is 2 X 2 .
"""

# Step 2 - Pooling
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding a second convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding a third convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding a forth convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding Dropout
classifier.add(tf.keras.layers.Dropout(0.2))

#step 3 -Flattening 
"""
1- in this step we are going to take the pooled feature map and convert it to a huge single vector.
"""
classifier.add(tf.keras.layers.Flatten())
#Step 4 -Full Connection
"""
- Dense has multiple arguments : 
	1- output_dim -> 100 is a good choice but a good practice 2^() so we use 128 as output nodes.
	2-activation function -> in  hidden layer we use (relu function ) .
	3- we use the softmax in the output layer to give the probabilities to each class.
"""
# Step 4 - Full Connection
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
classifier.add(tf.keras.layers.Dense(units=26, activation='softmax'))

#Compile the CNN
"""
- Comile method has multiple parameters: 
    - Optimizer is like a stochastic gradient descent to find the optimal number of the weights.
    - loss function.
"""
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set
history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size = 64, epochs = 50, verbose=0)
000
# Evaluating the accuracy
accuracy = classifier.evaluate(X_test, y_test)
print("Accuracy: ", accuracy[1])

# Saving the CNN Model in h5 file 
classifier.save('CNN_Model3.h5')

# Predicting Test set results
y_pred = classifier.predict(X_test)
classifier.summary()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
