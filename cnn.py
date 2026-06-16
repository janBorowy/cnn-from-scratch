import enum

import numpy as np


DEFAULT_LR = 0.1

class CNN:

    def __init__(self, layers, number_of_classes: int, learning_rate: float):
        self.layers = layers
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate

    def train(self, image: np.ndarray, label: int) -> tuple[float, bool]:
        data = (image / 255) - 0.5

        for layer in self.layers:
            data = layer.forward(data)

        is_correct = np.argmax(data) == label
        clipped_data = np.clip(data, 1e-15, 1.0)
        loss = -np.log(clipped_data[label])

        loss_gradient = np.zeros(self.number_of_classes)
        loss_gradient[label] = -1 / clipped_data[label]

        for layer in self.layers[::-1]:
            loss_gradient = layer.backprop(loss_gradient, self.learning_rate)

        return loss, is_correct

    def test(self, image: np.ndarray) -> tuple[float, bool]:
        return 0, False

# TODO: add stride and padding
# right now assumed padding = 0, stride = 1
# TODO: add bias
class Conv2D:

    def __init__(self, num_filters: int, input_depth: int, kernel_size: int=3):
        self.num_filters: int = num_filters
        self.kernel_size: int = kernel_size 

        self.filters: np.ndarray = np.random.randn(num_filters, kernel_size, kernel_size, input_depth) / kernel_size**2

    def __iterate_regions(self, data: np.ndarray):
        h, w, _ = data.shape
        k = self.kernel_size
        for i in range(h - k + 1):
            for j in range(w - k + 1):
                im_region = data[i:(i + k), j:(j + k), :]
                yield im_region, i, j
    
    def forward(self, data: np.ndarray):

        self.__cached_input = data

        height, width, _ = data.shape
        out_h = height - self.kernel_size + 1
        out_w = width - self.kernel_size + 1
        output = np.zeros((out_h, out_w, self.num_filters))

        for region, i, j in self.__iterate_regions(data):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2, 3))

        return output

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_input = np.zeros(self.__cached_input.shape)

        for region, i, j in self.__iterate_regions(self.__cached_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * region

                d_L_d_input[i:i+self.kernel_size, j:j+self.kernel_size] +=(
                    d_L_d_out[i, j, f] * self.filters[f]
                )

        self.filters -= learning_rate * d_L_d_filters

        return d_L_d_input

class ReLU:
    def forward(self, data: np.ndarray):
        self.__cached_input = data
        return np.maximum(0, data)

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):
        return d_L_d_out * (self.__cached_input > 0)

class MaxPool2D:

    def __init__(self, kernel_size: int):
        self.kernel_size: int = kernel_size

    def __iterate_regions(self, data: np.ndarray):
        h, w, _ = data.shape
        k = self.kernel_size
        h_out = h // k
        w_out = w // k

        for i in range(h_out):
            for j in range(w_out):
                im_region = data[(i * k):(i * k + k), (j * k):(j * k + k), :]
                yield im_region, i, j
        
    def forward(self, data: np.ndarray):
        self.__cached_data = data
        k = self.kernel_size
        height, width, num_filters = data.shape
        output = np.zeros((height // k, width // k, num_filters))

        for region, i, j in self.__iterate_regions(data):
            output[i, j] = np.amax(region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):

        d_L_d_input = np.zeros(self.__cached_data.shape)

        for region, i, j in self.__iterate_regions(self.__cached_data):
            height, width, depth = region.shape
            amax = np.amax(region, axis=(0, 1))
            
            for x in range(width):
                for y in range(height):
                    for z in range(depth):
                        if (region[x, y, z] == amax[z]):
                            d_L_d_input[i * 2 + x, j * 2 + y, z] = d_L_d_out[i, j, z]

        return d_L_d_input

class FCLayer:

    def __init__(self, input_len: int, output_len: int):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)

    def forward(self, data: np.ndarray):
        if (data.ndim > 1):
            data = data.flatten()

        return np.dot(data, self.weights) + self.biases

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):
        # TODO
        return 0

class Softmax:

    def __init__(self, input_len: int, output_len: int):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)

    def forward(self, data: np.ndarray):
        self.__cached_data_shape = data.shape

        if (data.ndim > 1):
            data = data.flatten()
        self.__cached_flat_input = data

        totals = np.dot(data, self.weights) + self.biases

        shifted = totals - np.max(totals)
        exp = np.exp(shifted)
        self.__cached_exp = exp
        self.__cached_exp_sum = np.sum(exp, axis=0)
        return exp / self.__cached_exp_sum

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):
        i = np.argmax(d_L_d_out != 0)
        val = d_L_d_out[i]
        t_exp = self.__cached_exp
        totals_sum = self.__cached_exp_sum

        d_out_d_t = -t_exp[i] * t_exp / totals_sum**2
        d_out_d_t[i] = t_exp[i] * (totals_sum - t_exp[i]) / (totals_sum ** 2)

        d_t_d_w = self.__cached_flat_input
        d_t_d_b = 1
        d_t_d_input = self.weights

        d_L_d_t = val * d_out_d_t
        d_L_d_w = d_t_d_w[np.newaxis, :].T @ d_L_d_t[np.newaxis, :]
        d_L_d_b = d_L_d_t * d_t_d_b
        d_L_d_input = d_t_d_input @ d_L_d_t

        self.weights -= learning_rate * d_L_d_w 
        self.biases -= learning_rate * d_L_d_b 
        return d_L_d_input.reshape(self.__cached_data_shape)


