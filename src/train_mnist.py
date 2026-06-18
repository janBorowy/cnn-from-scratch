from cv2.detail import TEST_CUSTOM
import numpy as np
import struct
import os
from cnn import CNN, Conv2D, MaxPool2D, Softmax
TRAIN_IMAGES=1000
TEST_IMAGES=100

def read_idx_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} in image file"
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows, cols)


def read_idx_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} in label file"
        return np.frombuffer(f.read(), dtype=np.uint8)


DATA_DIR = "./mnist"

train_images = read_idx_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"))[:TRAIN_IMAGES]
train_labels = read_idx_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"))[:TRAIN_IMAGES]
test_images  = read_idx_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"))[:TEST_IMAGES]
test_labels  = read_idx_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"))[:TEST_IMAGES]

cnn = CNN([
    Conv2D(8, 1, kernel_size=3),
    MaxPool2D(2),
    Softmax(13 * 13 * 8, 10)
], number_of_classes=10, learning_rate=0.005)

print(f"{'-'*5} Dataset initialized {'-'*5}")
print(f"Dataset size = {len(train_images) + len(test_images)}")
print(f'Train = {TRAIN_IMAGES}')
print(f'Test = {TEST_IMAGES}')

print(f"{'-'*5} Training started {'-'*5}")
for epoch in range(3):
    print(f"{'-'*5} Epoch {epoch+1} {'-'*5}")

    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0
        l, acc = cnn.train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print(f"{'-'*5} Testing start {'-'*5}")
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    predicted_label = cnn.test(im)
    num_correct += 1 if predicted_label == label else 0

num_tests = len(test_images)
print('Loss:', loss / num_tests)
print('Accuracy:', num_correct / num_tests)
