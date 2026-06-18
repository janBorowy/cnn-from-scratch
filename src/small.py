from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

LEARNING_RATE=0.001

cnn = CNN([
    Conv2D(16, 3, kernel_size=3),    # 62x62x16
    ReLU(),
    MaxPool2D(2),                    # 31x31x16

    Conv2D(32, 16, kernel_size=3),   # 29x29x32
    ReLU(),
    MaxPool2D(2),                    # 14x14x32

    FCLayer(14 * 14 * 32, 128),
    ReLU(),
    Softmax(128, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'small')
