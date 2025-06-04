import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

class NeuralNetLayer:
    def __init__(self):
        self.gradient = None
        self.parameters = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearLayer(NeuralNetLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.w = np.random.randn(output_size, input_size) * 0.01
        self.b = np.random.randn(output_size) * 0.01
        self.cur_input = None
        self.parameters = [self.w, self.b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cur_input = x
        return x @ self.w.T + self.b

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        assert self.cur_input is not None, "Must call forward before backward"
        dw = gradient[:, :, None] @ self.cur_input[:, None, :]
        db = gradient
        self.gradient = [dw, db]
        return gradient @ self.w


class ReLULayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.gradient = np.where(x > 0, 1.0, 0.0)
        return np.maximum(0, x)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        assert self.gradient is not None, "Must call forward before backward"
        return gradient * self.gradient


class SoftmaxOutputLayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()
        self.cur_probs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_stable = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(x_stable)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        self.cur_probs = probs
        return probs

    def backward(self, target: np.ndarray) -> np.ndarray:
        assert self.cur_probs is not None, "Must call forward before backward"
        return self.cur_probs - target


class Conv2DLayer(NeuralNetLayer):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int or Tuple[int, int], stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        self.w = np.random.randn(out_channels, in_channels, *self.kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.parameters = [self.w, self.b]
        self.cur_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, _, in_height, in_width = x.shape

        out_height = (in_height + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (in_width + 2*self.padding - self.kernel_size[1]) // self.stride + 1

        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                              (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.tensordot(x_slice, self.w, axes=([1,2,3], [1,2,3])) + self.b
        
        self.cur_input = x
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        batch_size = gradient.shape[0]
        _, _, in_height, in_width = self.cur_input.shape
        _, _, out_height, out_width = gradient.shape
        
        dw = np.zeros_like(self.w)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(self.cur_input)

        if self.padding > 0:
            x_padded = np.pad(self.cur_input, ((0, 0), (0, 0), 
                             (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dx_padded = np.pad(dx, ((0, 0), (0, 0), 
                              (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = self.cur_input
            dx_padded = dx

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]

                dw += np.tensordot(gradient[:, :, i, j], x_slice, axes=([0], [0]))

                db += np.sum(gradient[:, :, i, j], axis=0)

                dx_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(
                    gradient[:, :, i, j], self.w, axes=([1], [0]))
        
        # Remove padding
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        self.gradient = [dw / batch_size, db / batch_size]
        return dx


class MaxPool2DLayer(NeuralNetLayer):
    def __init__(self, pool_size: int or Tuple[int, int] = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride
        self.cur_input = None
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = x.shape
        self.cur_input = x
        
        out_height = (height - self.pool_size[0]) // self.stride + 1
        out_width = (width - self.pool_size[1]) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(x)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride
                w_end = w_start + self.pool_size[1]
                
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(x_slice, axis=(2,3))

                max_vals = output[:, :, i, j][:, :, None, None]
                self.mask[:, :, h_start:h_end, w_start:w_end] = (x_slice == max_vals)
        
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        batch_size, channels, out_height, out_width = gradient.shape
        dx = np.zeros_like(self.cur_input)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride
                w_end = w_start + self.pool_size[1]

                grad_slice = gradient[:, :, i, j][:, :, None, None]
                dx[:, :, h_start:h_end, w_start:w_end] += grad_slice * self.mask[:, :, h_start:h_end, w_start:w_end]
        
        return dx


class FlattenLayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()
        self.original_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient.reshape(self.original_shape)

class CNN:
    def __init__(self, *args: List[NeuralNetLayer]):
        self.layers = args

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, target: np.ndarray):
        for layer in self.layers[::-1]:
            target = layer.backward(target)

    def save_model(self, filename):
        save_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'parameters') and layer.parameters:
                # Store each parameter with reduced precision
                for j, param in enumerate(layer.parameters):
                    save_dict[f'layer_{i}_param_{j}'] = param.astype(np.float32)
        np.savez_compressed(filename, **save_dict)
    
    def load_model(self, filename):
        loaded = np.load(filename, allow_pickle=True)
        param_count = 0
        
        for layer in self.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                # Load each parameter for this layer
                num_params = len(layer.parameters)
                layer.parameters = [
                    loaded[f'layer_{param_count}_param_{i}'] 
                    for i in range(num_params)
                ]
                param_count += 1


class Optimizer:
    def __init__(self, net: CNN):
        self.net = net

    def step(self):
        for layer in self.net.layers:
            if layer.parameters is not None and layer.gradient is not None:
                self.update(layer.parameters, layer.gradient)

    def update(self, params: List[np.ndarray], gradient: List[np.ndarray]):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    def __init__(self, net: CNN, lr: float = 0.01):
        super().__init__(net)
        self.lr = lr

    def update(self, params: List[np.ndarray], gradient: List[np.ndarray]):
        for p, g in zip(params, gradient):
            p -= self.lr * g.mean(axis=0)

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    return np.eye(num_classes)[labels]


def preprocess_data(df: pd.DataFrame, input_size: int = 28) -> Tuple[np.ndarray, np.ndarray]:
    x = df.drop(columns='label').values.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = df['label'].values
    x = x.reshape(-1, 1, input_size, input_size)  # Reshape for CNN
    y = one_hot_encode(y)
    return x, y


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))


def train(cnn: CNN, optimizer: Optimizer, 
          x_train: np.ndarray, y_train: np.ndarray,
          x_test: np.ndarray, y_test: np.ndarray,
          epochs: int = 10, batch_size: int = 32) -> dict:
    
    num_samples = x_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        correct = 0
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        batch_pbar = tqdm(range(num_batches), desc="Batches", leave=False)
        
        for batch in batch_pbar:

            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_samples)
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = cnn.forward(x_batch)

            loss = -np.sum(y_batch * np.log(predictions + 1e-10)) / batch_size
            epoch_loss += loss
            correct += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
            cnn.backward(y_batch)
            optimizer.step()
            batch_pbar.set_postfix({'batch_loss': loss})

        avg_loss = epoch_loss / num_batches
        train_acc = correct / num_samples

        test_predictions = cnn.forward(x_test)
        test_acc = accuracy(test_predictions, y_test)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        epoch_pbar.set_postfix({
            'train_loss': f"{avg_loss:.4f}",
            'train_acc': f"{train_acc:.4f}",
            'test_acc': f"{test_acc:.4f}"
        })
    
    return history