from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

NUMBER_OF_CLASSES = 10
LEARNING_RATE = 0.001

cnn = CNN([
    Conv2D(32, 3, kernel_size=3, padding=1),      # 64x64x32
    ReLU(),
    MaxPool2D(2),                                 # 32x32x32

    Conv2D(64, 32, kernel_size=3, padding=1),     # 32x32x64
    ReLU(),
    MaxPool2D(2),                                 # 16x16x64

    Conv2D(128, 64, kernel_size=3, padding=1),    # 16x16x128
    ReLU(),
    MaxPool2D(2),                                 # 8x8x128

    Conv2D(256, 128, kernel_size=3, padding=1),   # 8x8x256
    ReLU(),
    MaxPool2D(2),                                 # 4x4x256

    FCLayer(4 * 4 * 256, 512),
    ReLU(),
    FCLayer(512, 128),
    ReLU(),
    Softmax(128, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'big')
