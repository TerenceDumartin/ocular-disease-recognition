
from tensorflow.keras import layers
from tensorflow.keras import models


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
