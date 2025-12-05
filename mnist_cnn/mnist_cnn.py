from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

class ConvLayer:
    def __init__(self, num_filters, filter_size, num_channels, stride=1, padding=0, learning_rate=0.01):
        """
        Initialize a convolutional layer.
        
        Args:
            num_filters: number of filters (output channels)
            filter_size: size of square filter (e.g., 3 for 3x3)
            num_channels: number of input channels
            stride: stride for convolution
            padding: padding size
            learning_rate: learning rate for weight updates
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
        
        self.w1 = np.random.randn(num_filters, num_channels, filter_size, filter_size) * 0.01
        self.b1 = np.zeros((1, num_filters))
    
    def forward(self, X):
        """
        Forward pass for convolution.
        
        Args:
            X: input tensor (batch_size, num_channels, height, width)
        
        Returns:
            output: (batch_size, num_filters, out_height, out_width)
        """
        
        # Step 1: Save input for backward pass
        self.X = X
        
        # Extract dimensions from input
        batch_size, num_channels, H, W = X.shape
        
        # Step 2: Add padding to input if needed
        if self.padding > 0:
            X_padded = np.pad(X, 
                             pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                             mode='constant', 
                             constant_values=0)
        else:
            X_padded = X
        
        # Save padded input for backward pass
        self.X_padded = X_padded
        
        # Get padded dimensions
        _, _, H_padded, W_padded = X_padded.shape
        
        # Calculate output dimensions
        out_H = ((H_padded - self.filter_size) // self.stride) + 1
        out_W = ((W_padded - self.filter_size) // self.stride) + 1
        
        # Step 3: Initialize output tensor
        output = np.zeros((batch_size, self.num_filters, out_H, out_W))
        
        # Step 4: Convolve each image in the batch with each filter
        for b in range(batch_size):
            image = X_padded[b]
            
            for f in range(self.num_filters):
                kernel = self.w1[f]
                conv_output = np.zeros((out_H, out_W))
                
                for h in range(out_H):
                    for w in range(out_W):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        image_region = image[:, 
                                            h_start:h_start + self.filter_size,
                                            w_start:w_start + self.filter_size]
                        
                        conv_output[h, w] = np.sum(image_region * kernel)
                
                output[b, f] = conv_output + self.b1[0, f]
        
        return output
    
    def backward(self, dout, learning_rate=None):
        """
        Backward pass for convolution layer.
        
        Args:
            dout: gradient of loss w.r.t. output (batch_size, num_filters, out_height, out_width)
            learning_rate: override default learning rate (optional)
        
        Returns:
            dX: gradient w.r.t. input (batch_size, num_channels, height, width)
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        batch_size, num_filters, out_H, out_W = dout.shape
        _, _, H_padded, W_padded = self.X_padded.shape
        
        # Initialize gradients
        dX_padded = np.zeros_like(self.X_padded)
        dW = np.zeros_like(self.w1)
        db = np.zeros_like(self.b1)
        
        # Step 1: Backpropagate through each batch and filter
        for b in range(batch_size):
            image = self.X_padded[b]
            
            for f in range(self.num_filters):
                # Get current filter and its gradient
                kernel = self.w1[f]
                grad_out = dout[b, f]  # (out_H, out_W)
                
                # Step 2: For each output position, backprop to input and weights
                for h in range(out_H):
                    for w in range(out_W):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # Get the input region that contributed to this output
                        image_region = image[:, 
                                           h_start:h_start + self.filter_size,
                                           w_start:w_start + self.filter_size]
                        
                        # Gradient of loss w.r.t. this output position
                        grad = grad_out[h, w]
                        
                        # Step 3: Compute gradient w.r.t. weights
                        # dL/dW = (dL/dout) * (dout/dW) = grad * image_region
                        dW[f] += grad * image_region
                        
                        # Step 4: Compute gradient w.r.t. input (backprop to previous layer)
                        # dL/dX = (dL/dout) * (dout/dX) = grad * kernel
                        dX_padded[b, :, 
                                 h_start:h_start + self.filter_size,
                                 w_start:w_start + self.filter_size] += grad * kernel
                
                # Step 5: Accumulate bias gradients
                # Bias is added to every output position, so gradient is sum of all output gradients
                db[0, f] += np.sum(grad_out)
        
        # Step 6: Remove padding from input gradient
        if self.padding > 0:
            dX = dX_padded[:, :, 
                          self.padding:self.padding + self.X.shape[2],
                          self.padding:self.padding + self.X.shape[3]]
        else:
            dX = dX_padded
        
        # Step 7: Update weights and biases using gradient descent
        self.w1 -= learning_rate * dW
        self.b1 -= learning_rate * db
        
        # Step 8: Return gradient w.r.t. input for previous layer
        return dX

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize max pooling layer.
        
        Args:
            pool_size: size of pooling window (e.g., 2 for 2x2)
            stride: stride for pooling (how many pixels to move each step)
        """
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, X):
        """
        Forward pass for max pooling.
        
        Args:
            X: input (batch_size, channels, height, width)
        
        Returns:
            output: pooled tensor (batch_size, channels, out_height, out_width)
        """
        
        # Extract dimensions from input
        batch_size, num_channels, H, W = X.shape
        
        # Step 1: Calculate output dimensions
        # Max pooling uses the same formula as convolution for output size
        # output_size = ((input_size - pool_size) / stride) + 1
        out_H = ((H - self.pool_size) // self.stride) + 1
        out_W = ((W - self.pool_size) // self.stride) + 1
        
        # Step 2: Initialize output tensor
        # Shape: (batch_size, channels, out_height, out_width)
        # Important: max pooling doesn't change the number of channels
        # It only reduces spatial dimensions
        output = np.zeros((batch_size, num_channels, out_H, out_W))
        
        # Optional: Save indices of max values for backward pass
        # This stores WHERE each max value came from
        self.max_indices = np.zeros((batch_size, num_channels, out_H, out_W, 2), dtype=int)
        
        # Step 3: Slide pooling window across input and take max values
        # Loop over batch
        for b in range(batch_size):
            # Loop over channels
            for c in range(num_channels):
                # Get the current image channel: (H, W)
                channel = X[b, c]
                
                # Slide pooling window across spatial dimensions
                for h in range(out_H):
                    for w in range(out_W):
                        # Calculate starting positions based on stride
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # Extract the pooling window region
                        # Shape: (pool_size, pool_size)
                        pool_region = channel[h_start:h_start + self.pool_size,
                                             w_start:w_start + self.pool_size]
                        
                        # Find the maximum value in this window
                        max_value = np.max(pool_region)
                        
                        # Store the max value in output
                        output[b, c, h, w] = max_value
                        
                        # Save the location (indices) of the max value for backward pass
                        # Flatten the pool_region to find where max is
                        flat_indices = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        
                        # Convert local indices (within pool_region) to global indices
                        # in the original channel
                        global_h = h_start + flat_indices[0]
                        global_w = w_start + flat_indices[1]
                        
                        # Store these global indices
                        self.max_indices[b, c, h, w] = [global_h, global_w]
        
        # Step 4: Save input for backward pass
        self.X = X
        
        # Step 5: Return the pooled output
        # Shape: (batch_size, channels, out_height, out_width)
        return output
    
    def backward(self, dout):
        """
        Backward pass for max pooling.
        
        Args:
            dout: gradient of loss w.r.t. output (batch_size, channels, out_height, out_width)
        
        Returns:
            dX: gradient w.r.t. input (batch_size, channels, height, width)
        """
        
        # Get dimensions
        batch_size, num_channels, out_H, out_W = dout.shape
        batch_size_X, num_channels_X, H, W = self.X.shape
        
        # Initialize gradient w.r.t. input (same shape as original input)
        dX = np.zeros_like(self.X)
        
        # Backpropagate gradients to input
        # Gradient only flows back to the position where max value came from
        for b in range(batch_size):
            for c in range(num_channels):
                for h in range(out_H):
                    for w in range(out_W):
                        # Get the global indices of where the max came from
                        max_h, max_w = self.max_indices[b, c, h, w]
                        
                        # Route the gradient back to that position
                        # All other positions in that pool_region get 0 gradient
                        dX[b, c, max_h, max_w] += dout[b, c, h, w]
        
        return dX
    

class ReLU:
    """ReLU activation function"""
    def forward(self, X):
        """Apply ReLU: max(0, x)"""
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dout):
        """Gradient: 1 if x > 0, else 0"""
        return dout * (self.X > 0)


class Flatten:
    """Flatten multi-dimensional tensor to 1D"""
    def forward(self, X):
        """
        Flatten: (batch_size, channels, height, width) -> (batch_size, -1)
        """
        self.shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, dout):
        """Reshape gradient back to original shape"""
        return dout.reshape(self.shape)


class FullyConnectedLayer:
    """Fully connected (dense) layer"""
    def __init__(self, input_size, output_size, learning_rate=0.01):
        """
        Initialize FC layer.
        
        Args:
            input_size: number of input features
            output_size: number of output classes
            learning_rate: learning rate for weight updates
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Forward pass: output = X @ W + b
        
        Args:
            X: (batch_size, input_size)
        
        Returns:
            output: (batch_size, output_size)
        """
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dout):
        """
        Backward pass through FC layer.
        
        Args:
            dout: gradient of loss w.r.t. output (batch_size, output_size)
        
        Returns:
            dX: gradient w.r.t. input
        """
        # Gradient w.r.t. input
        dX = dout @ self.W.T
        
        # Gradient w.r.t. weights
        dW = self.X.T @ dout
        
        # Gradient w.r.t. bias
        db = np.sum(dout, axis=0, keepdims=True)
        
        # Update weights using gradient descent
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        
        return dX


class SimpleCNN:
    def __init__(self, learning_rate=0.01):
        """
        Build a simple CNN for MNIST:
        - Input: 28x28 grayscale images
        - Conv layer: 8 filters, 3x3, padding=1
        - ReLU activation
        - MaxPool: 2x2, stride=2
        - Flatten
        - FC layer: -> 10 classes (digits 0-9)
        """
        # Step 1: Initialize convolutional layer
        # Input: (batch, 1, 28, 28)
        # Output: (batch, 8, 28, 28) - padding=1 keeps spatial dims same
        self.conv = ConvLayer(num_filters=8, 
                             filter_size=3, 
                             num_channels=1,
                             stride=1, 
                             padding=1)
        
        # Step 2: ReLU activation
        self.relu = ReLU()
        
        # Step 3: Max pooling layer
        # Input: (batch, 8, 28, 28)
        # Output: (batch, 8, 14, 14) - reduces spatial dimensions by 2
        self.pool = MaxPoolLayer(pool_size=2, stride=2)
        
        # Step 4: Flatten layer
        # Input: (batch, 8, 14, 14) = (batch, 1568)
        # Output: (batch, 1568)
        self.flatten = Flatten()
        
        # Step 5: Fully connected layer
        # Input: 1568 features
        # Output: 10 classes (digits 0-9)
        self.fc = FullyConnectedLayer(input_size=8*14*14, 
                                     output_size=10,
                                     learning_rate=learning_rate)
        
        self.learning_rate = learning_rate
    
    def forward(self, X):
        """
        Forward pass through entire network.
        
        Args:
            X: input images (batch_size, 1, 28, 28) or (batch_size, 28, 28) for MNIST
        
        Returns:
            output: class logits (batch_size, 10)
        """
        # Ensure input has channel dimension
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]  # Add channel dimension
        
        # Pass through each layer sequentially
        x = self.conv.forward(X)      # Conv: (batch, 1, 28, 28) -> (batch, 8, 28, 28)
        x = self.relu.forward(x)      # ReLU: element-wise max(0, x)
        x = self.pool.forward(x)      # MaxPool: (batch, 8, 28, 28) -> (batch, 8, 14, 14)
        x = self.flatten.forward(x)   # Flatten: (batch, 8, 14, 14) -> (batch, 1568)
        x = self.fc.forward(x)        # FC: (batch, 1568) -> (batch, 10)
        
        return x
    
    def softmax(self, X):
        """Softmax normalization for probabilities"""
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def compute_loss(self, logits, y):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: raw network output (batch_size, 10)
            y: true labels (batch_size,) - integers 0-9
        
        Returns:
            loss: scalar cross-entropy loss
        """
        batch_size = logits.shape[0]
        
        # Convert logits to probabilities
        probs = self.softmax(logits)
        
        # Cross-entropy loss: -log(P(correct_class))
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        loss = -np.mean(np.log(probs[np.arange(batch_size), y]))
        
        return loss
    
    def backward(self, logits, y):
        """
        Backward pass through entire network.
        
        Args:
            logits: raw network output (batch_size, 10)
            y: true labels (batch_size,)
        """
        batch_size = logits.shape[0]
        
        # Step 1: Compute gradient of loss w.r.t. logits
        # For cross-entropy with softmax: gradient is (probs - one_hot)
        probs = self.softmax(logits)
        dout = probs.copy()
        dout[np.arange(batch_size), y] -= 1
        dout /= batch_size
        
        # Step 2: Backpropagate through FC layer
        dout = self.fc.backward(dout)
        
        # Step 3: Backpropagate through flatten
        dout = self.flatten.backward(dout)
        
        # Step 4: Backpropagate through max pool
        # Note: This is simplified - true backprop through pool uses saved max indices
        dout = self.pool.backward(dout)  # Placeholder - would need full implementation
        
        # Step 5: Backpropagate through ReLU
        dout = self.relu.backward(dout)
        
        # Step 6: Backpropagate through Conv
        dout = self.conv.backward(dout)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Training loop.
        
        Args:
            X_train: training images (N, 28, 28)
            y_train: training labels (N,) - integers 0-9
            X_val: validation images (optional) for monitoring generalization
            y_val: validation labels (optional)
            epochs: number of training epochs
            batch_size: batch size for training
        """
        num_samples = X_train.shape[0]
        
        print(f"\n{'='*70}")
        print(f"Starting SimpleCNN model training: {num_samples} samples, {epochs} epochs, batch_size={batch_size}")
        print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            # Step 1: Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Step 2: Process in mini-batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Step 3: Forward pass
                logits = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(logits, y_batch)
                epoch_loss += loss
                num_batches += 1
                
                # Step 4: Backward pass and weight updates
                self.backward(logits, y_batch)
                
                # Print batch progress every 5 batches
                if (batch_start // batch_size + 1) % 5 == 0:
                    avg_batch_loss = epoch_loss / num_batches
                    num_batches_total = (num_samples + batch_size - 1) // batch_size
                    current_batch = (batch_start // batch_size) + 1
                    print(f"  Batch {current_batch}/{num_batches_total}, Avg Loss: {avg_batch_loss:.4f}")
            
            # Step 5: Print epoch progress
            avg_loss = epoch_loss / num_batches
            train_acc = self.evaluate(X_train, y_train)
            
            output = f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}"
            
            # If validation data provided, evaluate on it too
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                output += f" | Val Accuracy: {val_acc:.4f}"
            
            print(output)
        
        print(f"\n{'='*70}")
        print(f"SimpleCNN model training complete")
        print(f"{'='*70}\n")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: input images (N, 28, 28)
        
        Returns:
            predictions: class indices (N,)
        """
        logits = self.forward(X)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate model accuracy on a dataset.
        
        Args:
            X: input images (N, 1, 28, 28) or (N, 28, 28)
            y: true labels (N,) - integers 0-9
        
        Returns:
            accuracy: fraction of correct predictions (0.0 to 1.0)
        """
        # Step 1: Make predictions on the entire dataset
        predictions = self.predict(X)
        
        # Step 2: Compare predictions with true labels
        # Create boolean array: True where prediction matches label
        correct = (predictions == y)
        
        # Step 3: Calculate accuracy as fraction of correct predictions
        accuracy = np.mean(correct)
        
        return accuracy

cnn = SimpleCNN()

X_train_small = X_test[:5000]
y_train_small = y_test[:5000]
cnn.train(X_train_small, y_train_small, epochs=5, batch_size=32)

# Use a small subset for faster testing during development
X_test_small = X_test[:1500]
y_test_small = y_test[:1500]
accuracy = cnn.evaluate(X_test_small, y_test_small)
print(f"SimpleCNN model accuracy: {accuracy:.4f}")

