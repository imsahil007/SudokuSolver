from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten

import time
import glob
import torch
import os

def buildlenetmodel():
    # BUILD CONVOLUTIONAL NEURAL NETWORKS
   
    model = Sequential()

    model.add(Conv2D(32, (5,5) , activation='relu', input_shape = (28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(50, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense( 9 ))
    model.add(Activation('softmax'))
  
    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model

def checkGPU():
    print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

