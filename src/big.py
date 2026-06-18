from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

NUMBER_OF_CLASSES = 10
LEARNING_RATE = 0.001

cnn = CNN([
    Conv2D(32, 3, kernel_size=3),      # 222x222x32
    ReLU(),
    MaxPool2D(2),                      # 111x111x32

    Conv2D(64, 32, kernel_size=3),     # 109x109x64
    ReLU(),
    MaxPool2D(2),                      # 54x54x64

    Conv2D(128, 64, kernel_size=3),    # 52x52x128
    ReLU(),
    MaxPool2D(2),                      # 26x26x128

    Conv2D(256, 128, kernel_size=3),   # 24x24x256
    ReLU(),
    MaxPool2D(2),                      # 12x12x256

    FCLayer(12 * 12 * 256, 512),
    ReLU(),
    FCLayer(512, 128),
    ReLU(),
    Softmax(128, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'big', 5)
