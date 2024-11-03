import tensorflow as tf

def preprocess(file_path):
    
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

    image = img_generator.flow_from_directory(
        file_path,
        target_size=(512, 512),
        batch_size=16,
        class_mode=None,
        shuffle=False,
        seed=40
    )
    
    return image

def get_prediction(input_path):
    """Takes the folder path as input and returns the predictions with the input data"""
    
    input_data = preprocess(input_path)
    
    model = tf.keras.models.load_model('model/real2comic_V1.h5')
    input_data = next(input_data)
    predictions = model.predict(input_data)
    
    return predictions, input_data