from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

NUMBER_OF_CLASSES = 10
LEARNING_RATE = 0.001

cnn = CNN([
    Conv2D(32, 3, kernel_size=3),      # 62x62x32
    ReLU(),
    MaxPool2D(2),                      # 31x31x32

    Conv2D(64, 32, kernel_size=3),     # 29x29x64
    ReLU(),
    MaxPool2D(2),                      # 14x14x64

    Conv2D(128, 64, kernel_size=3),    # 12x12x128
    ReLU(),
    MaxPool2D(2),                      # 6x6x128

    FCLayer(6 * 6 * 128, 512),
    ReLU(),
    FCLayer(512, 128),
    Softmax(128, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'medium')
