from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


class MaskDetector:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        return Sequential([
            Conv2D(64, 7, activation='relu', padding='same', input_shape=(32, 32, 3)),
            MaxPooling2D(2),
            Conv2D(128, 3, activation='relu', padding='same'),
            MaxPooling2D(2),
            Conv2D(256, 3, activation='relu', padding='same'),
            MaxPooling2D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
