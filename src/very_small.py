from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

LEARNING_RATE=0.001

cnn = CNN([
    Conv2D(8, 3, kernel_size=3),     # 62x62x8
    ReLU(),
    MaxPool2D(2),                    # 31x31x8

    Conv2D(16, 8, kernel_size=3),    # 29x29x16
    ReLU(),
    MaxPool2D(2),                    # 14x14x16

    Softmax(14 * 14 * 16, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'very_small')
