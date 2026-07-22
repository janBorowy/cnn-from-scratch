# cnn-from-scratch

A convolutional neural network implemented **from scratch in pure NumPy** — no PyTorch, no TensorFlow. Each layer (`Conv2D`, `MaxPool2D`, `ReLU`, `FCLayer`, `Softmax`) implements its own forward pass and manual backpropagation, and layers can be freely composed into custom architectures.

The repo ships with four pre-built architectures (`very_small`, `small`, `medium`, `big`) that classify 10 bird species from a subset of [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), plus a minimal MNIST example.

## Project structure

```
src/
├── cnn.py            # Core library: CNN container + Conv2D, ReLU, MaxPool2D, FCLayer, Softmax
├── evaluate.py        # Dataset loading + train_and_test() training/eval loop
├── very_small.py       # Pre-built architectures
├── small.py
├── medium.py
├── big.py
├── test_model.py      # Loads a trained .pkl model and evaluates it
└── train_mnist.py     # Standalone MNIST example
models/                 # Pickled trained models
CUB_200_2011/            # Training/test images, one folder per class
own_training_data/      # Extra images per class, used by test_model.py
run_trainings.sh         # Trains all 4 architectures in parallel via tmux
run_tests.sh              # Tests all 4 architectures in parallel via tmux
```

## Installation

```bash
git clone https://github.com/janBorowy/cnn-from-scratch.git
cd cnn-from-scratch

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scripts must be run from the project root — they read from `./CUB_200_2011` and `./own_training_data` and write to `./models`, `./results`, `./images`.

## Running the pre-built architectures

| Architecture | Conv layers | Filters | FC layers |
|---|---|---|---|
| `very_small` | 2 | 8 → 16 | none |
| `small` | 2 | 16 → 32 | 1 |
| `medium` | 3 | 32 → 64 → 128 | 2 |
| `big` | 4 | 32 → 64 → 128 → 256 | 2 |

**Train** an architecture (trains for 15 epochs, saves the best epoch's model to `models/<name>.pkl`):

```bash
python3 src/small.py   # or very_small.py / medium.py / big.py
```

**Test** a trained model (prints accuracy and saves a confusion matrix to `images/`):

```bash
python3 src/test_model.py small
```

**Run all four at once** (requires `tmux`):

```bash
./run_trainings.sh
./run_tests.sh
```

**MNIST example** (requires raw IDX files under `./mnist/`):

```bash
python3 src/train_mnist.py
```

## Building your own architecture

A network is just a `CNN` wrapping an ordered list of layers:

```python
from cnn import CNN, Conv2D, ReLU, MaxPool2D, FCLayer, Softmax

cnn = CNN(
    layers=[...],
    number_of_classes=10,
    learning_rate=0.001
)
```

- `cnn.train(image, label) -> (loss, is_correct)` — forward pass, cross-entropy loss, backprop, weight updates.
- `cnn.test(image) -> predicted_label` — forward pass only.

### Available layers

| Layer | Constructor | Notes |
|---|---|---|
| `Conv2D` | `Conv2D(num_filters, input_depth, kernel_size=3, padding=0, stride=1)` | `input_depth` must match the previous layer's output channels. |
| `ReLU` | `ReLU()` | No parameters. |
| `MaxPool2D` | `MaxPool2D(kernel_size)` | Non-overlapping pooling; divides spatial dims by `kernel_size`. |
| `FCLayer` | `FCLayer(input_len, output_len)` | Flattens its input automatically. |
| `Softmax` | `Softmax(input_len, output_len)` | Use as the final layer; `output_len` should equal `number_of_classes`. |

### Example: writing a new architecture script

Create a new file in `src/`, following the pattern used by `small.py`/`medium.py`/etc.:

```python
# src/my_arch.py
from cnn import *
from evaluate import NUMBER_OF_CLASSES, train_and_test

cnn = CNN([
    Conv2D(16, 3, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),

    Conv2D(32, 16, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),

    FCLayer(16 * 16 * 32, 256),   # must match the flattened output of the last conv/pool layer
    ReLU(),
    Softmax(256, NUMBER_OF_CLASSES)
], number_of_classes=NUMBER_OF_CLASSES, learning_rate=0.001)

if __name__ == '__main__':
    train_and_test(cnn, 'my_arch')
```

Run and evaluate it the same way as any built-in architecture:

```bash
python3 src/my_arch.py
python3 src/test_model.py my_arch
```

**Sizing tips:** a `Conv2D` with no padding shrinks height/width by `kernel_size - 1`; with `padding = (kernel_size - 1) // 2` it preserves spatial size. `MaxPool2D(k)` divides height/width by `k`. The first `FCLayer`/`Softmax` after the conv stack needs `input_len = height * width * channels` of the last layer's output, and the first `Conv2D`'s `input_depth` must match your input image's channel count (3 for RGB, 1 for grayscale).
