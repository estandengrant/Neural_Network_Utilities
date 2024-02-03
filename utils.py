import numpy as np
import os
import sys
import math

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # To prevent overflow as a result of exponential function
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CCE(Loss):
    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)
        y_predictions_clip = np.clip(y_prediction, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_conf = y_predictions_clip[range(samples), y_true]
        else:
            correct_conf = np.sum(y_predictions_clip*y_true, axis=1)
        neg_log_likeihood = -np.log(correct_conf)
        return neg_log_likeihood

'''

Example of how functions above can be used to contruct Network and perform forward propogation

Layer 1:
dense_1 = DenseLayer(2, 3)
activation_1 = ActivationReLU()

Output Layer:
dense_2 = DenseLayer(3, 3)
activation_2 = ActivationSoftMax()


# Run Model
dense_1.forward(X)
activation_1.forward(dense_1.output)

dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)

loss_func = CCE()
loss = loss_func.calculate(activation_2.output, y)

# print(activation_2.output[:5])
print("loss: ", loss)
'''
