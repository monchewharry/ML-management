from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# model = Sequential()


def set_model(params: dict, input_shape: tuple):
    m = Sequential()
    m.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    m.add(Conv2D(64, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(params['num_classes'], activation='softmax'))

    m.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return m
