import numpy as np
from copy import deepcopy

DEFAULT_LR = 0.001

class CNN:

    def __init__(self, layers, number_of_classes: int, learning_rate: float):
        self.layers = layers
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate

    # return cross-entropy loss and if was correct
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

    # returns predicted class label
    def test(self, image: np.ndarray) -> int:
        data = (image / 255) - 0.5

        for layer in self.layers:
            data = layer.forward(data)

        return int(np.argmax(data))
    
class Conv2D:

    def __init__(self, num_filters: int, input_depth: int, kernel_size: int=3, padding: int=0, stride: int=1):
        self.num_filters: int = num_filters
        self.kernel_size: int = kernel_size 
        self.stride = stride
        self.padding = padding

        self.filters: np.ndarray = np.random.randn(num_filters, kernel_size, kernel_size, input_depth) \
            * np.sqrt(2.0 / (kernel_size**2 * input_depth))
        self.biases: np.ndarray = np.zeros(num_filters)

    def __pad(self, data: np.ndarray) -> np.ndarray:
        p = self.padding
        if p == 0:
            return data
        return np.pad(data, ((p, p), (p, p), (0, 0)), mode='constant')

    def __iterate_regions(self, data: np.ndarray):
        h, w, _ = data.shape
        k = self.kernel_size
        s = self.stride

        for i in range((h - k) // s + 1):
            for j in range((w - k) // s + 1):
                im_region = data[i*s : i*s + k, j*s : j*s + k, :]
                yield im_region, i, j
    
    def forward(self, data: np.ndarray):

        if data.ndim ==2:
            data = data[..., np.newaxis]

        data = self.__pad(data)
        self.__cached_input = data

        height, width = 0, 0
        if (len(data.shape) == 2):
            height, width = data.shape
        elif (len(data.shape) == 3):
            height, width, _ = data.shape

        k = self.kernel_size
        s = self.stride

        out_h = (height - k) // s + 1
        out_w = (width - k) // s + 1
        output = np.zeros((out_h, out_w, self.num_filters))

        for region, i, j in self.__iterate_regions(data):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2, 3)) + self.biases

        return output

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):
        d_L_d_f = np.zeros(self.filters.shape)
        d_L_d_b = np.zeros(self.biases.shape)
        d_L_d_input = np.zeros(self.__cached_input.shape)

        k = self.kernel_size
        s = self.stride

        for region, i, j in self.__iterate_regions(self.__cached_input):
            for f in range(self.num_filters):
                d_L_d_f[f] += d_L_d_out[i, j, f] * region
                d_L_d_b[f] += d_L_d_out[i, j, f]

                d_L_d_input[i*s:i*s+k, j*s:j*s+k, :] +=(
                    d_L_d_out[i, j, f] * self.filters[f]
                )

        self.filters -= learning_rate * d_L_d_f
        self.biases -= learning_rate * d_L_d_b

        p = self.padding
        if p > 0:
            d_L_d_input = d_L_d_input[p:-p, p:-p, :]

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
        h, w = 0, 0
        if (len(data.shape) == 2):
            h, w = data.shape
        elif (len(data.shape) == 3):
            h, w, _ = data.shape

        k = self.kernel_size
        h_out = h // k
        w_out = w // k

        for i in range(h_out):
            for j in range(w_out):
                im_region = data[(i * k):(i * k + k), (j * k):(j * k + k)]
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
            k = self.kernel_size
            height, width, depth = region.shape
            amax = np.amax(region, axis=(0, 1))
            
            for y in range(height):
                for x in range(width):
                    for z in range(depth):
                        if (region[y, x, z] == amax[z]):
                            d_L_d_input[i * k + y, j * k + x, z] = d_L_d_out[i, j, z]

        return d_L_d_input

class FCLayer:

    def __init__(self, input_len: int, output_len: int):
        self.weights = np.random.randn(input_len, output_len) * np.sqrt(2.0 / input_len)
        self.biases = np.zeros(output_len)

    def forward(self, data: np.ndarray):
        self.__cached_data_shape = data.shape
        if (data.ndim > 1):
            data = data.flatten()
        self.__cached_data = data
        return np.dot(data, self.weights) + self.biases

    def backprop(self, d_L_d_out: np.ndarray, learning_rate: float):
        d_t_d_w = self.__cached_data
        d_t_d_b = 1
        d_t_d_input = self.weights

        d_L_d_w = d_t_d_w[np.newaxis, :].T @ d_L_d_out[np.newaxis, :]
        d_L_d_b = d_L_d_out * d_t_d_b
        d_L_d_input = d_t_d_input @ d_L_d_out

        self.weights -= learning_rate * d_L_d_w
        self.biases  -= learning_rate * d_L_d_b

        return d_L_d_input.reshape(self.__cached_data_shape)

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


