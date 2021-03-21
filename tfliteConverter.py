#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:44:58 2021

@author: eman-saeed
"""



"""
TensorFlow Lite is a set of tools to help developers run TensorFlow models on mobile,
and IoT devices. 
It enables on-device machine learning inference with low latency and a small binary size.

this line of codes helps us to make model applicable on mobile application by converting the  generated weights from cnn model to tflite file . 

"""

import tensorflow as tf
tflite_model = tf.keras.models.load_model('CNN_Model3.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
tflite_save = converter.convert()
open("generated.tflite", "wb").write(tflite_save)