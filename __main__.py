import os
import cv2
import numpy as np
from random import shuffle
from math import floor

from cnn import CNN, Conv2D, MaxPool2D, Softmax, FCLayer, ReLU

DATA_DIR="./CUB_200_2011/CUB_200_2011/images"
NUMBER_OF_CLASSES=10
TRAIN_PERCENTAGE=80
INPUT_SIZE=(224, 224)
LEARNING_RATE=0.1

paths = os.listdir(DATA_DIR)
paths.sort()

train_images: list[np.ndarray] = []
train_labels: list[int] = []
test_images: list[np.ndarray] = []
test_labels: list[int] = []
label_to_class_name: list[str] = []

print(f'Train = 80%')
print(f'Test = {100 - TRAIN_PERCENTAGE}%')
print(f'Starting training for {NUMBER_OF_CLASSES} classes')

for class_name in paths[:NUMBER_OF_CLASSES]:
    print(f"--- Loading class {class_name} ---")

    image_paths = os.listdir(os.path.join(DATA_DIR, class_name))
    shuffle(image_paths)
    train_images_num = floor(TRAIN_PERCENTAGE/100 * len(image_paths))

    print(f"Total images # = {len(image_paths)}")
    print(f"Train images # = {train_images_num}")
    print(f"Test images # = {len(image_paths) - train_images_num}")

    for image in image_paths[:train_images_num]:
        data = cv2.imread(os.path.join(DATA_DIR, class_name, image))
        data = cv2.resize(data, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)
        train_images.append(data)
        train_labels.append(len(label_to_class_name))

    for image in image_paths[train_images_num:]:
        data = cv2.imread(os.path.join(DATA_DIR, class_name, image))
        data = cv2.resize(data, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)
        test_images.append(data)
        test_labels.append(len(label_to_class_name))

    label_to_class_name.append(class_name)

cnn = CNN([
    Conv2D(32, 3, kernel_size=3),   # 222x222x32
    ReLU(),
    Conv2D(32, 32, kernel_size=3),  # 220x220x32
    ReLU(),
    MaxPool2D(2),                    # 110x110x32

    Conv2D(64, 32, kernel_size=3),  # 108x108x64
    ReLU(),
    Conv2D(64, 64, kernel_size=3),  # 106x106x64
    ReLU(),
    MaxPool2D(2),                    # 53x53x64

    Conv2D(128, 64, kernel_size=3), # 51x51x128
    ReLU(),
    MaxPool2D(2),                    # 25x25x128  (floor division)

    Softmax(25 * 25 * 128, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=LEARNING_RATE)

total_loss = 0
num_correct_all = 0
num_correct = 0

all_images_shuffled = list(zip(test_images + train_images, test_labels + train_labels))
shuffle(all_images_shuffled)

for i, (image, label) in enumerate(all_images_shuffled):
    loss, was_correct = cnn.train(image, label)
    total_loss += loss
    num_correct += 1 if was_correct else 0
    num_correct_all += 1 if was_correct else 0
    if i % 10 == 9:
        print(f'[Step {i + 1}] Past 10 steps: Average loss: {(total_loss / 100):.3f} | Accuracy: {(num_correct / 10) * 100:.0f}%')
        total_loss = 0
        num_correct = 0

print(f"Average accuracy total: {num_correct_all * 100 / (len(test_images) + len(train_images)):.0f}%")

