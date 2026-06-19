import os
import sys
from pathlib import Path
import pickle
import cv2
import time
import numpy as np
from random import shuffle, seed
import matplotlib.pyplot as plt

from cnn import CNN

NUMBER_OF_CLASSES=10
MODELS_DIR = "./models"
DATA_DIR = "./CUB_200_2011"
OWN_DATA_DIR= "./own_training_data"
INPUT_SIZE=(64, 64)

if len(sys.argv) < 2:
    print("Missing model name argument")
    exit(1)

model_name = sys.argv[1]

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
print("")

print(f"{'-'*5} Shuffled dataset contents {'-'*5}")
print(f"Shuffling test images")
shuffelled_dataset = list(zip(dataset_labels, dataset_images))
shuffle(shuffelled_dataset)
shuffeled_labels, shuffeled_images = zip(*shuffelled_dataset)
for i, class_name in enumerate(label_to_class_name):
    print(f"{class_name}: {len([x for x in shuffeled_labels if x == i])}")
print("")

own_dataset_labels = []
own_dataset_images = []
for i, class_name in enumerate(label_to_class_name):
    print(f"{'-'*5} Loading own training data for class {class_name} {'-'*5}")
    images_path = os.listdir(os.path.join(OWN_DATA_DIR, class_name))
    print(f"Total images # = {len(images_path)}")

    for image in images_path:
        data = cv2.imread(os.path.join(OWN_DATA_DIR, class_name, image))
        data = cv2.resize(data, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)
        own_dataset_images.append(data)
        own_dataset_labels.append(i)

test_labels = list(shuffeled_labels) + own_dataset_labels
test_images = list(shuffeled_images) + own_dataset_images

print(f"{'-'*5} Final test dataset {'-'*5}")
for i, class_name in enumerate(label_to_class_name):
    print(f"{class_name}: {len([x for x in test_labels if x == i])}")
print("")

print(f"{'-'*5} Loading model {model_name} from {MODELS_DIR}/{model_name}.pkl {'-'*5}")
with open(f"{MODELS_DIR}/{model_name}.pkl", "rb") as fh:
    model: CNN = pickle.load(fh)

print(f"{'-'*5} Test start {'-'*5}")
num_correct = 0
num_correct_own = 0
predicted = []

start = time.time_ns()
for i, (label, image) in enumerate(zip(test_labels, test_images)):
    predicted_label = model.test(image)
    predicted.append(predicted_label)
    was_correct = predicted_label == label
    num_correct += 1 if was_correct else 0
    if i >= len(shuffeled_labels):
        num_correct_own += 1 if was_correct else 0
    print(f"[Step {i + 1}] Predicted: {label_to_class_name[predicted_label]} | Actual: {label_to_class_name[label]} | Hit?={'YES!' if was_correct else 'no'}")
testing_time = time.time_ns() - start
print(f"Predcited {num_correct}/{len(test_labels)}")
print(f"Test accuracy: {num_correct/len(test_labels) * 100:.0f}%")
print(f"Test accuracy on own data: {num_correct_own/(len(test_labels) - len(shuffeled_labels)) * 100:.0f}%")
print(f"Time taken to test: {testing_time / 1_000_000_000:.2f} seconds")

confusion_matrix = []
for i, _ in enumerate(label_to_class_name):
    confusion_matrix.append([0 for _ in range(NUMBER_OF_CLASSES)])
    true_positions = [pos for pos, label in enumerate(test_labels) if label == i]
    for j in true_positions:
        confusion_matrix[i][predicted[j]] += 1

labels = [
    "001.Black_footed_Albatross", "002.Laysan_Albatross", "003.Sooty_Albatross",
    "004.Groove_billed_Ani", "005.Crested_Auklet", "006.Least_Auklet",
    "007.Parakeet_Auklet", "008.Rhinoceros_Auklet", "009.Brewer_Blackbird",
    "010.Red_winged_Blackbird"
]

cm_array = np.array(confusion_matrix)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm_array, cmap="Blues")

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(NUMBER_OF_CLASSES))
ax.set_yticks(range(NUMBER_OF_CLASSES))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
ax.set_ylabel("True", fontsize=12, labelpad=10)
ax.set_title("Confusion Matrix", fontsize=14, pad=15)

thresh = cm_array.max() / 2
for i in range(cm_array.shape[0]):
    for j in range(cm_array.shape[1]):
        ax.text(j, i, cm_array[i, j],
                ha="center", va="center", fontsize=9,
                color="white" if cm_array[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(f"./images/{model_name}_confusion_matrix.png")
plt.show()

