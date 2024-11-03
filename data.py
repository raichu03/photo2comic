import tensorflow as tf

def paired_generator(face_gen, comic_gen):
    """Autoencoder needs paired images, input and label, so generating the paired version"""
    while True:
        # Get a batch from each generator
        face_images = face_gen.next()
        comic_images = comic_gen.next()
        
        # Yield the batch of paired images
        yield (face_images, comic_images)

def data_preprocess(image_path):
    """This function preprocesses the images and labels before feeding it to the label"""
    
    ### Initialiaing the data generator to preporcess the data ###
    img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
    )
    
    processed_image = img_generator.flow_from_directory(
    image_path,
    target_size=(512, 512),
    batch_size=16,
    class_mode=None,
    shuffle=False,
    seed=40
    )
    
    return processed_image

def get_data(train_input: str, train_label: str, validation_input: str, validation_lable: str):
    """Returns the train and validation data"""
    
    train_face = data_preprocess(train_input)
    train_comic = data_preprocess(train_label)
    validation_face = data_preprocess(validation_input)
    validation_comic = data_preprocess(validation_lable)
    
    train_generator = paired_generator(train_face, train_comic)
    val_generator = paired_generator(validation_face, validation_comic)
    
    return train_generator, val_generator