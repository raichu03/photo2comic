################# ALL THE IMPROTS ###################
import os
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential

from keras.layers.reshaping.reshape import Reshape
######################################################


############# TO LOAD THE TRAINING AND VALIDATION DATA IN BATCHES ##############
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
###############################################################################


################### TRAINING IMAGES #########################
face_train = img_generator.flow_from_directory('data/train_images',target_size=(512,512),shuffle=False,
                                               seed=40,save_format='jpg', batch_size=16, classes=None,
                                               class_mode=None)

comic_train = img_generator.flow_from_directory('data/train_cartoon',target_size=(512,512),shuffle=False,
                                                seed=40,save_format='jpg',batch_size=16, classes=None,
                                                class_mode=None)


################## VALADITION IMAGES ########################
face_val = img_generator.flow_from_directory('data/val_images',target_size=(512,512),shuffle=False,
                                             seed=40,save_format='jpg',batch_size=4,classes=None,
                                             class_mode=None)

comic_val = img_generator.flow_from_directory('data/val_cartoon',target_size=(512,512),shuffle=False,
                                              seed=40,save_format='jpg',batch_size=4,classes=None,
                                              class_mode=None)
####################################################################################################


############### TO FEED THE DATA IN model.fit ############
train_generator=zip(face_train, comic_train)
val_generator=zip(face_val, comic_val)
##########################################################



###################### AUTOENCODER MODEL ###############################
model=Sequential(name="real2comic")
model.add(keras.Input(shape=(512,512,3)))
model.add(layers.Conv2D(3,3,activation='relu', padding='same'))
model.add(layers.Conv2D(32,3,activation='relu',strides=2, padding='same'))
model.add(layers.Conv2D(32,3,activation='relu',strides=2, padding='same'))
model.add(layers.Conv2D(32,3,activation='relu',strides=2, padding='same'))

model.add(layers.Conv2D(32,3,activation='relu',padding='same'))


model.add(layers.Conv2DTranspose(32,3,activation='relu', padding='same'))
model.add(layers.Conv2DTranspose(32,3,activation='relu', strides=2, padding='same'))
model.add(layers.Conv2DTranspose(32,3,activation='relu', strides=2, padding='same'))
model.add(layers.Conv2DTranspose(3,3, activation='sigmoid',strides=2, padding='same'))

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

model.summary()

#########################################################################


######################### TRAINING THE MODEL ########################
model.fit(x=train_generator, validation_data = val_generator, steps_per_epoch=500,
          validation_steps=500, epochs=50)
##############################################################################


####################### SAVING THE MODEL ###########################
model.save("modelV1.h5")
#################################################################