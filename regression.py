import numpy as np
import math

class LogisticRegression:
    
    def __init__(self, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.linear = None
        self.accuracy = None
        self.y_true = None
    

    # train model for evaluation
    def fit(self, x, y):
        self.y_true = y
        n_samples, n_features = x.shape

        # weights and biases
        # works something like this: 
        # f(x) = ∑(w • x) + b

        # perform matrix multiplication with the weights and the inputs
        # then add the bias at the end

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = self._sigmoid(np.dot(x, self.weights) + self.bias)

            # gradient decent

            y_w = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            y_b = (1 / n_samples) * np.sum(y_pred - y)

            # update weights and biases

            self.weights -= self.lr * y_w[0]
            self.weights -= self.lr * y_w[1]
            self.bias -= self.lr * y_b


    def predict(self, x):
        # store linear output
        self.linear = np.dot(x, self.weights) + self.bias
        # overall prediction
        prediction = self._sigmoid(self.linear)
        # model accuracy
        self.accuracy = self._accuracy(self.y_true, prediction)

        return prediction
    

    # accuracy measurement algorithm
    def _accuracy(self, y_true, y_pred):
        self.accuracy = np.sum(y_true == y_pred) / len(y_true)

        return self.accuracy
    

    def _swish(self, x):
        return x / (1 + np.exp(-x))
    

    def _relu(self, x):
        return max(0.0, x)


    # sigmoid function returns a probability between 0 and 1
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def _softplus(self, x, limit=30):
        if x >= limit:
            return x
        return math.log(1 + np.exp(x), np.exp(1))
    

    def _softmax(self, x, theta=1.0, axis=None):
        e_x = np.exp(x)
        sum_e_x = np.sum(np.exp(x), axis=0)

        return e_x / sum_e_x
    

    # Mean Squared Error cost function
    def _MSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)