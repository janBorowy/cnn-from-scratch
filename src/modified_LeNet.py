from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

LEARNING_RATE=0.001

cnn = CNN([
    Conv2D(6, 3, kernel_size=5),    # 220x220x6
    ReLU(),
    MaxPool2D(2),                    # 55x55x6

    Conv2D(16, 6, kernel_size=5),   # 51x51x16
    ReLU(),
    MaxPool2D(2),                    # 27x27x16

    FCLayer(27 * 27 * 16, 84),
    ReLU(),
    Softmax(84, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)


