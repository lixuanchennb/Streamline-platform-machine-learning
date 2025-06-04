from func import NeuralNetLayer, LinearLayer, ReLULayer, SoftmaxOutputLayer, MaxPool2DLayer, Conv2DLayer, FlattenLayer, CNN, GradientDescentOptimizer, one_hot_encode, train, preprocess_data
from data import load_mnist_data_as_dataframe
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history: dict):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    train_df, test_df = load_mnist_data_as_dataframe()
    input_size = 28
    output_size = 10
    
    if train_df is not None and test_df is not None:
        x_train, y_train = preprocess_data(train_df)
        x_test, y_test = preprocess_data(test_df)
        
        # Special Structure specially desigend for Master. Wu Yutian
        cnn_layers = [
            Conv2DLayer(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            ReLULayer(),
            MaxPool2DLayer(pool_size=2, stride=2),
            Conv2DLayer(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ReLULayer(),
            MaxPool2DLayer(pool_size=2, stride=2),
            FlattenLayer(),
            LinearLayer(input_size=64*(input_size//4)*(input_size//4), output_size=128),
            ReLULayer(),
            LinearLayer(input_size=128, output_size=output_size),
            SoftmaxOutputLayer()
        ]
        
        cnn = CNN(*cnn_layers)
        
        # Create optimizer
        learning_rate = 0.02
        optimizer = GradientDescentOptimizer(cnn, learning_rate)
        
        # Train the model
        epochs = 5
        batch_size = 64

        plot_training_history(history = train(cnn, optimizer, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size))
        # Save the model
        cnn.save_model("mnist_cnn_model.npz")
        print("Model trained and saved successfully.")

        # #load the model
        # cnn.load_model("mnist_cnn_model.npz")
        # print("Model loaded successfully.")

