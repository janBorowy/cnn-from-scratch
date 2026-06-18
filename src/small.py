from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

LEARNING_RATE=0.001

cnn = CNN([
    Conv2D(16, 3, kernel_size=3),    # 222x222x16
    ReLU(),
    MaxPool2D(2),                    # 111x111x16

    Conv2D(32, 16, kernel_size=3),   # 109x109x32
    ReLU(),
    MaxPool2D(2),                    # 54x54x32

    Conv2D(64, 32, kernel_size=3),   # 52x52x64
    ReLU(),
    MaxPool2D(4),                    # 13x13x64

    FCLayer(13 * 13 * 64, 128),
    ReLU(),
    Softmax(32, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'small', 8)
