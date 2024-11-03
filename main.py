import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential

import data

def get_model():
    ###################### AUTOENCODER MODEL ###############################
    model=Sequential(name="real2comic")
    model.add(keras.Input(shape=(512,512,3)))
    model.add(layers.Conv2D(3,3,activation='relu', padding='same'))
    model.add(layers.Conv2D(32,3,activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), padding='same'))
    model.add(layers.Conv2D(64,3,activation='relu', padding='same'))
    model.add(layers.Conv2D(64,3,activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), padding='same'))
    model.add(layers.Conv2D(128,3,activation='relu', padding='same'))
    model.add(layers.Conv2D(128,3,activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), padding='same'))

    model.add(layers.Conv2D(128,3,activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(128,3,activation='relu', padding='same'))

    model.add(layers.Conv2DTranspose(128,3,activation='relu',strides=2, padding='same'))
    model.add(layers.Conv2DTranspose(64,3,activation='relu',strides=2, padding='same'))
    model.add(layers.Conv2DTranspose(32,3,activation='relu',strides=2, padding='same'))
    model.add(layers.Conv2DTranspose(3,3,activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    return model

def main():
    
    ### TODO: Add the file path for each ###
    train_input = ''
    train_label = ''
    val_input = ''
    val_label = ''
    
    train_generator, val_generator = data.get_data(train_input, train_label, val_input, val_label)
    
    model = get_model()
    
    model.fit(x=train_generator, validation_data = val_generator,steps_per_epoch=530, validation_steps=94 ,epochs=35)
    
    ### TODO: Change the name of the model as your need ###
    model.save("model_name.h5")


if __name__ == '__main__':
    main()