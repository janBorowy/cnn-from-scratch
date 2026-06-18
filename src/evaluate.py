import os
import cv2
import numpy as np
import pickle
from random import shuffle
from math import floor
from cnn import CNN
from pathlib import Path
from datetime import datetime
from copy import deepcopy

MAX_EPOCHS=15
NUMBER_OF_CLASSES=10
DATA_DIR="./CUB_200_2011/CUB_200_2011/images"
MODELS_DIR="./models"
TRAIN_PERCENTAGE=90
INPUT_SIZE=(64, 64)

print(f"{'-'*5} Preparing dataset {'-'*5}")
print(f"Data directory: {Path(DATA_DIR).resolve()}")
paths = os.listdir(DATA_DIR)
paths.sort()

dataset_images: list[np.ndarray] = []
dataset_labels: list[int] = []
label_to_class_name: list[str] = []

for class_name in paths[:NUMBER_OF_CLASSES]:
    print(f"{'-'*5} Loading class {class_name} {'-'*5}")
    images_path = os.listdir(os.path.join(DATA_DIR, class_name))
    print(f"Total images # = {len(images_path)}")

    for image in images_path:
        data = cv2.imread(os.path.join(DATA_DIR, class_name, image))
        data = cv2.resize(data, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)
        dataset_images.append(data)
        dataset_labels.append(len(label_to_class_name))

    label_to_class_name.append(class_name)


train_dataset_size = floor(len(dataset_images) * TRAIN_PERCENTAGE / 100)

print(f"{'-'*5} Dataset initialized {'-'*5}")
print(f"Dataset size = {len(dataset_images)}")
print(f'Train = {train_dataset_size} ({TRAIN_PERCENTAGE}%)')
print(f'Test = {len(dataset_images) - train_dataset_size} ({100 - TRAIN_PERCENTAGE}%)')

shuffelled_dataset = list(zip(dataset_labels, dataset_images))
shuffle(shuffelled_dataset)

shuffeled_labels, shuffeled_images = zip(*shuffelled_dataset)

train_labels = shuffeled_labels[:train_dataset_size]
train_images = shuffeled_images[:train_dataset_size]
test_labels = shuffeled_labels[train_dataset_size:]
test_images = shuffeled_images[train_dataset_size:]

def augment(image: np.ndarray) -> np.ndarray:
    if np.random.rand() > 0.5:
        image = np.fliplr(image)

    factor = np.random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.uint8)

    return image

def log(s: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]{s}",)

def train_and_test(cnn: CNN, model_name: str):
    print(f"{'-'*5} Training started for model {model_name} {'-'*5}")
    accuracies = []
    models = []

    for epoch in range(MAX_EPOCHS):
        print(f"{'-'*5} Epoch {epoch+1} {'-'*5}")
        total_loss = 0
        train_num_correct = 0

        epoch_data = list(zip(train_labels, train_images))
        shuffle(epoch_data)

        for i, (label, image) in enumerate(epoch_data):
            loss, was_correct = cnn.train(augment(image), label)
            total_loss += loss
            train_num_correct += 1 if was_correct else 0

            log(f'[Step {i + 1}] Loss: {loss:.2f} | Was correct: {was_correct}')

        num_correct = 0
        for i, (label, image) in enumerate(zip(test_labels, test_images)):
            predicted_label = cnn.test(image)
            num_correct += 1 if predicted_label == label else 0
            log(f"[Test step {i + 1}/{len(test_labels)}] Predicted: {label_to_class_name[predicted_label]} | Actual: {label_to_class_name[label]} | Hit?={'YES!' if predicted_label == label else 'no'}")

        models.append(deepcopy(cnn))
        accuracies.append(num_correct / len(test_labels)) 
        log(f"[Epoch {epoch+1}] Overall error rate: {(1 - (train_num_correct + num_correct) / (len(train_labels) + len(test_labels))) * 100:.0f}%")
        log(f"[Epoch {epoch+1}] Test accuracy: {num_correct / len(test_labels) * 100:.0f}%")

    best_accuracy_epoch, _ = max(enumerate(accuracies), key=lambda x: x[1])
    model_path = f'{MODELS_DIR}/{model_name}.pkl'
    with open(model_path, 'wb') as fh:
        pickle.dump(models[best_accuracy_epoch], fh, pickle.HIGHEST_PROTOCOL)

    log(f"Saved model from epoch {best_accuracy_epoch + 1} to {model_path}")

    cnn = models[best_accuracy_epoch]
    print(f"{'-'*5} Testing start {'-'*5}")
    num_correct = 0
    for i, (label, image) in enumerate(zip(test_labels, test_images)):
        predicted_label = cnn.test(image)
        num_correct += 1 if predicted_label == label else 0
        log(f"[Test step {i + 1}] Predicted: {label_to_class_name[predicted_label]} | Actual: {label_to_class_name[label]} | Hit?={'YES!' if predicted_label == label else 'no'}")

    print(f"{'-'*5} Testing complete {'-'*5}")
    log(f"Accuracy acquired: {num_correct / len(test_labels) * 100:.0f}%")

