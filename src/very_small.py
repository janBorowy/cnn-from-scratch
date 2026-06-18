from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

LEARNING_RATE=0.001

cnn = CNN([
    Conv2D(8, 3, kernel_size=3),     # 222x222x8
    ReLU(),
    MaxPool2D(4),                    # 55x55x8

    Conv2D(16, 8, kernel_size=3),    # 53x53x16
    ReLU(),
    MaxPool2D(4),                    # 13x13x16

    Softmax(13 * 13 * 16, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'very_small', 15)
