from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_addons as tfa
from tensorflow.keras.applications.vgg16 import VGG16


#---------------------------------------------------
#   BASICS MODELS
#---------------------------------------------------
def get_model_binary():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), input_shape=(256, 256, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    ### Flattening
    model.add(layers.Flatten())
    ### One fully connected
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy', 
            optimizer='adam',
            metrics=['accuracy'])

    return model


def get_model_classifier(out_shape):
        
    kappa=tfa.metrics.CohenKappa(num_classes=out_shape)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), input_shape=(256, 256, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    ### Flattening
    model.add(layers.Flatten())
    ### One fully connected
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(out_shape, activation='softmax'))


    model.compile(loss='categorical_crossentropy', 
            optimizer='adam',
            metrics=[kappa])

    return model

#---------------------------------------------------
#   TRANSFERT LEARNING
#---------------------------------------------------
def load_model_vgg16(in_shape):
    model = VGG16(weights="imagenet", include_top=False, input_shape=in_shape)  # shape = X_train[0].shape
    return model

# def add_last_layers_classif(model, out_shape):
#     model.trainable = False
#     flatten_layer = layers.Flatten()
#     dense_layer = layers.Dense(200, activation='relu')
#     prediction_layer = layers.Dense(out_shape, activation='softmax')
#     model = models.Sequential([
#         model,
#         flatten_layer,
#         dense_layer,
#         prediction_layer
#     ])
    
#     return model

# def compile_model_classif(model, out_shape):
#     kappa=tfa.metrics.CohenKappa(num_classes=out_shape)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=[kappa])
    
#     return model


def get_model_classifier_vgg16(in_shape, out_shape):
        
    model = load_model_vgg16(in_shape)
    
    model.trainable = False
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(200, activation='relu')
    prediction_layer = layers.Dense(out_shape, activation='softmax')
    model = models.Sequential([
        model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])
    kappa=tfa.metrics.CohenKappa(num_classes=out_shape)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[kappa])
    
    return model


def get_model_vgg16(in_shape):
    model = load_model_vgg16(in_shape)
    
    model.trainable = False
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(200, activation='relu')
    prediction_layer = layers.Dense(1, activation='sigmoid')
    model = models.Sequential([
        model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model