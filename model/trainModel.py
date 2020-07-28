

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

# from processData import processData
# from model import buildlenetmodel,checkGPU

PATH ="/data/"


X,Y=  processData(PATH)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
print("Training size: "+ str(len(X_train)))
print("Test size: "+ str(len(X_test)))
Y_train = to_categorical(Y_train, num_classes = 9)
Y_test = to_categorical(Y_test, num_classes = 9)
print('Shapes:')
print('Xtrain', str(X_train.shape))
print('Y_train', str(Y_train.shape))
print('X_test', str(X_test.shape))
print('Y_test', str(Y_test.shape))

datagenerated = ImageDataGenerator(
    shear_range=0.2,
    zoom_range = 0.2,
    width_shift_range=0.3, 
    height_shift_range=0.3)
model = buildlenetmodel()
checkGPU()

# DECREASE LEARNING RATE EACH EPOCH
# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
#We can use this. But I am using checkpoints instead to get the best of it
# TRAIN NETWORKS
History = None
epochs = 100

# Checkpoint for saving lowest loss model
cp_1 = ModelCheckpoint('lowest_loss_model', monitor = 'val_loss', mode = 'min',
                    save_best_only = True, verbose = 1)

cp_2 = ModelCheckpoint('best_accuracy_model', monitor = 'val_accuracy', mode = 'max',
                    save_best_only = True, verbose = 1)


History = model.fit(datagenerated.flow(X_train,Y_train, batch_size=16),
                                    epochs = epochs, steps_per_epoch = len(X_train)/16,callbacks = [cp_1,cp_2],
                                    validation_data = (X_test,Y_test), verbose=1)



