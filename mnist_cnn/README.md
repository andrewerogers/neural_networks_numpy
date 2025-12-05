# Building a Convolutional Neural Network from Scratch

## Introduction

In this activity, you'll build a complete CNN from scratch using only NumPy to classify images from the MNIST digits dataset. By the end, you'll understand:
- How convolution operations work
- Why pooling layers are useful
- How CNNs preserve spatial structure
- How to connect convolutional layers to fully-connected layers

**Dataset**: MNIST (28x28 grayscale images of handwritten digits 0-9)

You can load it with:
```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Normalize to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

---

## Part 1: Understanding Convolution

### 1.1 What is Convolution?

A convolution operation slides a small filter (kernel) across an image, computing dot products at each position. This detects local patterns like edges, corners, and textures.

**Your Task**: Implement a basic 2D convolution function.

```python
import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution on a single image with a single kernel.
    
    Args:
        image: (H, W) array
        kernel: (KH, KW) array
        stride: step size for sliding the kernel
        padding: number of pixels to pad around the image
    
    Returns:
        output: convolved feature map
    
    TODO: Implement this function
    Steps:
    1. Add padding to the image if padding > 0 (use np.pad)
    2. Calculate output dimensions
    3. Slide kernel across image, computing dot product at each position
    4. Return the output feature map
    """
    pass

# Test with a simple edge detection kernel
test_image = np.random.rand(5, 5)
edge_kernel = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]])
result = convolve2d(test_image, edge_kernel)
print(f"Input shape: {test_image.shape}, Output shape: {result.shape}")
```

**Hint**: Output height = (H + 2*padding - KH) // stride + 1

---

## Part 2: Building Convolution Layer

### 2.1 Forward Pass

A convolutional layer has multiple filters and can process batches of multi-channel images.

```python
class ConvLayer:
    def __init__(self, num_filters, filter_size, num_channels, stride=1, padding=0):
        """
        Initialize a convolutional layer.
        
        Args:
            num_filters: number of filters (output channels)
            filter_size: size of square filter (e.g., 3 for 3x3)
            num_channels: number of input channels
            stride: stride for convolution
            padding: padding size
        
        TODO: Initialize weights and biases
        - weights shape: (num_filters, num_channels, filter_size, filter_size)
        - biases shape: (num_filters,)
        - Use small random values for weights (e.g., * 0.01)
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        
        # Initialize parameters here
        pass
    
    def forward(self, X):
        """
        Forward pass for convolution.
        
        Args:
            X: input tensor (batch_size, channels, height, width)
        
        Returns:
            output: (batch_size, num_filters, out_height, out_width)
        
        TODO: Implement forward pass
        1. Save input for backward pass
        2. Add padding if needed
        3. For each image in batch, for each filter, convolve and add bias
        4. Return output
        """
        pass
```

---

## Part 3: Pooling Layer

### 3.1 Max Pooling

Pooling reduces spatial dimensions and provides translation invariance.

```python
class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize max pooling layer.
        
        Args:
            pool_size: size of pooling window
            stride: stride for pooling
        """
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, X):
        """
        Forward pass for max pooling.
        
        Args:
            X: input (batch_size, channels, height, width)
        
        Returns:
            output: pooled tensor
        
        TODO: Implement max pooling
        1. Calculate output dimensions
        2. For each pooling window, take the maximum value
        3. Save indices of max values for backward pass (optional for now)
        """
        pass
```

---

## Part 4: Activation Functions

### 4.1 ReLU

```python
class ReLU:
    def forward(self, X):
        """
        Apply ReLU activation: f(x) = max(0, x)
        
        TODO: Implement ReLU
        - Save input for backward pass
        - Return max(0, X)
        """
        pass
    
    def backward(self, dout):
        """
        Backward pass for ReLU.
        
        Args:
            dout: gradient from next layer
        
        Returns:
            dx: gradient with respect to input
        
        TODO: Implement backward pass
        - Gradient is 1 where input > 0, else 0
        """
        pass
```

---

## Part 5: Flatten Layer

This connects convolutional layers to fully-connected layers.

```python
class Flatten:
    def forward(self, X):
        """
        Flatten (batch, channels, height, width) to (batch, channels*height*width)
        
        TODO: Implement flattening
        - Save input shape for backward pass
        - Reshape to (batch_size, -1)
        """
        pass
    
    def backward(self, dout):
        """
        Reshape gradient back to original input shape
        """
        pass
```

---

## Part 6: Fully Connected Layer

You already know this one! It's the same as in your XOR network.

```python
class FullyConnectedLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        """
        TODO: Initialize weights and biases
        Same as your XOR network
        """
        pass
    
    def forward(self, X):
        """
        TODO: Implement forward pass
        output = X @ weights + bias
        """
        pass
    
    def backward(self, X, dout):
        """
        TODO: Implement backward pass and weight updates
        Similar to your XOR network
        """
        pass
```

---

## Part 7: Softmax and Cross-Entropy Loss

For multi-class classification, we use softmax activation and cross-entropy loss.

```python
def softmax(x):
    """
    Compute softmax activation.
    
    TODO: Implement numerically stable softmax
    Hint: exp(x - max(x)) / sum(exp(x - max(x)))
    """
    pass

def cross_entropy_loss(predictions, labels):
    """
    Compute cross-entropy loss.
    
    Args:
        predictions: softmax probabilities (batch_size, num_classes)
        labels: true class indices (batch_size,)
    
    Returns:
        loss: scalar loss value
    
    TODO: Implement cross-entropy
    loss = -mean(log(predictions[labels]))
    """
    pass
```

---

## Part 8: Building the Complete CNN

Now assemble everything into a simple CNN architecture:

**Architecture**: Conv -> ReLU -> MaxPool -> Flatten -> FC -> Softmax

```python
class SimpleCNN:
    def __init__(self):
        """
        Build a simple CNN for MNIST:
        - Conv layer: 8 filters, 3x3, padding=1
        - ReLU
        - MaxPool: 2x2
        - Flatten
        - FC layer: -> 10 classes
        
        TODO: Initialize all layers
        """
        pass
    
    def forward(self, X):
        """
        Forward pass through entire network.
        
        TODO: Pass input through each layer sequentially
        """
        pass
    
    def backward(self, X, y):
        """
        Backward pass through entire network.
        
        TODO: 
        1. Compute loss gradient
        2. Backpropagate through each layer in reverse order
        """
        pass
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Training loop.
        
        TODO:
        1. For each epoch, shuffle data
        2. Process in mini-batches
        3. Forward pass, compute loss
        4. Backward pass, update weights
        5. Print progress
        """
        pass
```

---

## Part 9: Backward Pass for Convolution (Challenge!)

This is the trickiest part. The backward pass for convolution involves:
- Computing gradients with respect to inputs
- Computing gradients with respect to filters

```python
# Add to ConvLayer class
def backward(self, dout, learning_rate=0.01):
    """
    Backward pass for convolution layer.
    
    Args:
        dout: gradient from next layer (batch, num_filters, out_h, out_w)
        learning_rate: learning rate for weight updates
    
    Returns:
        dx: gradient with respect to input
    
    TODO (Advanced):
    1. Compute dW (gradient w.r.t. filters) by convolving input with dout
    2. Compute db (gradient w.r.t. biases) by summing dout
    3. Compute dx (gradient w.r.t. input) by "full convolution" with flipped filters
    4. Update weights and biases
    
    This is complex! Consider starting with a simplified version or looking up 
    the mathematical derivation.
    """
    pass
```

---

## Part 10: Training and Evaluation

```python
# Prepare data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to (batch, channels, height, width)
X_train = X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0

# Use a small subset for faster training during development
X_train_small = X_train[:1000]
y_train_small = y_train[:1000]

# Create and train CNN
cnn = SimpleCNN()
cnn.train(X_train_small, y_train_small, epochs=5, batch_size=32)

# Evaluate
def evaluate(model, X, y):
    """
    TODO: Compute accuracy on test set
    """
    pass

accuracy = evaluate(cnn, X_test[:100], y_test[:100])
print(f"Test Accuracy: {accuracy:.2%}")
```

---

## Bonus Challenges

Once you have a working CNN:

1. **Add more layers**: Try Conv->ReLU->Conv->ReLU->MaxPool->Flatten->FC
2. **Implement dropout**: Randomly zero out activations during training
3. **Try different optimizers**: Add momentum or implement Adam optimizer
4. **Visualize filters**: Plot what your convolutional filters learn
5. **Data augmentation**: Add random rotations/translations to training data
6. **Batch normalization**: Normalize activations between layers

---

## Key Insights to Understand

As you implement this, make sure you understand:

- **Why convolution?** It detects local patterns and shares weights across spatial locations
- **Why pooling?** Reduces computation, provides translation invariance, grows receptive field
- **Why multiple filters?** Each filter learns different features (edges, textures, shapes)
- **Channel dimensions**: Input channels (RGB, grayscale), output channels (filters)
- **Spatial hierarchy**: Early layers detect simple features, deeper layers combine them

---

## Debugging Tips

- Start with tiny examples (3x3 image, 2x2 filter) and verify by hand
- Print shapes at every layer to catch dimension mismatches
- Implement forward pass fully before attempting backward pass
- Test each component independently before combining
- Use numerical gradient checking to verify your backward pass
- Start with small learning rates (0.001-0.01)

Good luck! This is challenging but incredibly rewarding. You'll truly understand CNNs once you've built one from scratch.