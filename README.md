# CNN_approach
   cnn approach consists of multiple files: 
    1- cnn.py 
    2- tfconverter.py
    3- cnn_model.h5
    4- generated.tflite.
   ## cnn.py
       cnn.py is a full cnn model that convert mniset kaggle dataset for sign language for a specific classes for each letter . 
       the accuracy of this model is 94 % .
   ## tfconverter.py
      TensorFlow Lite is a set of tools to help developers run TensorFlow models on mobile, embedded, and IoT devices. 
      It enables on-device machine learning inference with low latency and a small binary size.
      tfconverter.py is used for convert  h5 file to tflite that will be applicabel on a mobile phone . 
   ## cnn_model.h5 
      is the colected weights from cnn model.
   ## generated.tflite.
      is the result of conversion from h5 file to tflite file . 
