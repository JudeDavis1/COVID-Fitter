import math
import time
import numpy as np

class LogisticRegression:
    
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.linear = None
        self.accuracy = None
        self.y_true = None

        self.activation = self._relu
    

    # train model for evaluation
    def fit(self, x, y):
        self.y_true = y
        n_features = x.shape[0]

        # weights and biases
        # works something like this: 
        # f(x) = (w • x) + b

        # perform matrix multiplication with the weights and the inputs
        # then add the bias at the end

        self.weights = 0.01 * np.random.random(n_features)
        self.bias = 0.01 * np.random.random()

        y_w = np.array([])
        y_b = 0
        loss = float('inf')
        for i in range(self.epochs):
        # i = 0
        # while loss > 0.002:
            z = np.dot(x, self.weights) + self.bias
            y_pred = self.activation(z)

            # gradient decent
            y_w = x * self.activation(z, derivative=True) * self._MSE(self.y_true, y_pred, derivative=True)
            y_b = self.activation(z, derivative=True) * (self._MSE(self.y_true, y_pred, derivative=True))

            # update weights and biases

            self.weights += self.lr * y_w.flatten()
            self.bias += self.lr * y_b

            loss =  self._MSE(self.y_true, y_pred)
            print(f'Epoch {i + 1} Loss: {loss}')

            # time.sleep(0.01)
            # i += 1


    def predict(self, x):
        # store linear output
        self.linear = np.dot(x, self.weights) + self.bias
        # overall prediction
        prediction = self.activation(self.linear)
        # model accuracy
        self.accuracy = self._accuracy(self.y_true, prediction)

        return prediction
    

    # accuracy measurement algorithm
    def _accuracy(self, y_true, y_pred):
        self.accuracy = np.sum(np.round(y_true) == np.round(y_pred)) / len(y_true)

        return self.accuracy
    

    def _relu(self, x: np.ndarray, derivative=False):
        if derivative:
            return np.where(x <= 0, 0, 1)

        return np.maximum(0.0, x)
    
    def _leaky_relu(self, x: np.ndarray, alpha=0.3, derivative=False):
        if derivative:
            return np.where(x > 0, 1, alpha)
        
        return np.where(x > 0, x, x * alpha)


    # sigmoid function returns a probability between 0 and 1
    def _sigmoid(self, x, derivative=False):
        s = 1 / (1 + np.exp(-x))

        if derivative:
            return s * (1 - s)
        
        return s

    def _softmax(self, x, theta=1.0, axis=None):
        e_x = np.exp(x)
        sum_e_x = np.sum(np.exp(x), axis=0)

        return e_x / sum_e_x
    

    # Mean Squared Error cost function
    def _MSE(self, y_true, y_pred, derivative=False):
        if derivative:
            return 2 * (y_true - y_pred)
        return np.sum((y_true - y_pred)**2)