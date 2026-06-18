from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

NUMBER_OF_CLASSES = 10
LEARNING_RATE = 0.001

cnn = CNN([
    Conv2D(96, 3, kernel_size=11, stride=4),    # 54x54x96,
    ReLU(),                                     
    MaxPool2D(2),                               # 27x27x96
                                                
    Conv2D(256, 96, kernel_size=5, padding=2),  # 27x27x256
    ReLU(),                                     
    MaxPool2D(2),                               # 13x13x256
                                                
    Conv2D(384, 256, kernel_size=3, padding=1), # 13x13x384
    ReLU(),                                     
    Conv2D(384, 384, kernel_size=3, padding=1), # 13x13x384
    ReLU(),                                     
    Conv2D(256, 384, kernel_size=3, padding=1), # 13x13x384
    ReLU(),                                     
    MaxPool2D(2),                               # 6x6x256 

    FCLayer(6 * 6 * 256, 9216),
    ReLU(),
    FCLayer(9216, 4096),
    ReLU(),
    Softmax(4096, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    train_and_test(cnn, 'alex_net', 5)
