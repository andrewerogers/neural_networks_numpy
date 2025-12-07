import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
        """
        Simple feedforward neural network with one hidden layer.
        
        Args:
            input_size: Number of input neurons (2 for XOR)
            hidden_size: Number of hidden layer neurons
            output_size: Number of output neurons (1 for XOR)
            learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate
        
        # Initialize weights with small random values
        # - W1 dimensions (2, 4)
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        # - W2 dimensions (4, 1)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data (n_samples, input_size)
        
        Returns:
            Output predictions
        """
        # Hidden layer
        # - X dimensions: (n_samples, 2)
        # - X dot W1 dimensions: (n_samples, 2) @ (2, 4) = (n_samples, 4)
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        # - A1 dimensions: (n_samples, 4)
        # - W2 dimensions: (4, 1) 
        # - A1 dot W2 dimensions: (n_samples, 1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward propagation using MSE loss: L = 1/2 * (y - output)^2
        Computes dL/dW for each layer and updates weights via gradient descent.
        
        Args:
            X: Input data
            y: True labels
            output: Network predictions
        """
        m = X.shape[0]
        
        # dL/doutput = -(y - output) for MSE loss
        # But we compute (y - output) and add (gradient ascent on negative loss)
        # This is equivalent to gradient descent on loss
        # Chain rule: dL/dz2 = dL/doutput * doutput/dz2
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        # Backpropagate error to hidden layer
        # dL/da1 = dL/dz2 * dz2/da1 = output_delta * w2^T
        self.hidden_error = self.output_delta.dot(self.w2.T)
        # dL/dz1 = dL/da1 * da1/dz1
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)
        
        # Compute gradients and update weights
        # dL/dw2 = a1^T * dL/dz2
        self.w2 += self.a1.T.dot(self.output_delta) * self.lr
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.lr
        
        # dL/dw1 = X^T * dL/dz1
        self.w1 += X.T.dot(self.hidden_delta) * self.lr
        self.b1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.lr
    
    def train(self, X, y, epochs=10000):
        """
        Train the network using gradient descent.
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training iterations
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        output = self.forward(X)
        return (output > 0.5).astype(int)


# XOR problem dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1],
              [1, 0]])

y = np.array([[0],
              [1],
              [1],
              [0],
              [0]])

# Create and train the network
print("Training Neural Network on XOR Problem...")
print("-" * 40)
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
nn.train(X, y, epochs=10000)

# Test the network
print("\n" + "=" * 40)
print("Testing the trained network:")
print("=" * 40)
predictions = nn.predict(X)
outputs = nn.forward(X)

for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {predictions[i][0]} (Raw: {outputs[i][0]:.4f}), Expected: {y[i][0]}")