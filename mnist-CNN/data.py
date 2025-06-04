# Data Management for MNIST dataset
import numpy as np
import pandas as pd
import os
import struct

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist_data_as_dataframe():
    try:
        # Load training and test data
        file_path = 'mnist-CNN/mnist-70k'
        X_train = load_mnist_images(os.path.join(file_path, 'train-images-idx3-ubyte'))
        y_train = load_mnist_labels(os.path.join(file_path, 'train-labels-idx1-ubyte'))
        X_test = load_mnist_images(os.path.join(file_path, 't10k-images-idx3-ubyte'))
        y_test = load_mnist_labels(os.path.join(file_path, 't10k-labels-idx1-ubyte'))
        
        # Convert to DataFrame
        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test)
        test_df['label'] = y_test

        # use 10% of the training data for time saving
        train_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
      
        return train_df, test_df
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the MNIST data files are available in the 'data' directory.")
        return None, None

if __name__ == "__main__":
    train_df, test_df = load_mnist_data_as_dataframe()
    if train_df is not None and test_df is not None:
        print(f"Training DataFrame shape: {train_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")
        print(train_df.head())  # Display the first few rows of the training DataFrame
    else:
        print("Failed to load MNIST data.")