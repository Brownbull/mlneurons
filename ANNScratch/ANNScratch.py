# ANN from scratch based on 
# https://victorzhou.com/blog/intro-to-neural-networks/

import numpy as np

# Debug Flags
DebugNeuron = False
DebugNNtwrk = True

def sigmoid(x):
    # Our activation function:
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def mse_loss(y_test, y_pred):
    # y_test, y_pred are numpy arrays of same length
    return ((y_test - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias, costFunc):
        self.weights = weights
        self.bias = bias
        self.costFunc = costFunc

    def feedForward(self, inputs):
        dotWeights = np.dot(self.weights, inputs)
        addBias = dotWeights + self.bias
        result = self.costFunc(addBias)
        return result

    def debug(self, inputs, Wcount):
        dotWeights = np.dot(self.weights, inputs)
        addBias = dotWeights + self.bias
        accWXI = 0
        print("-"*15 + "> Weights Dot Product")
        for index, i in enumerate(self.weights):
            print("W%d" % (index + 1 + Wcount) + ": w-> " + "%.2f" % i + " * i->" +
                  "%.2f" % inputs[index], " = " + "%.2f" % inputs[index] * i + " + " +
                  str(accWXI) + " = " + "%.2f" % (accWXI + inputs[index] * i))
            accWXI += inputs[index] * i
        print("-"*15 + "> Bias Addition")
        print("%.2f" % dotWeights + " + " + str(self.bias) + " = " + "%.2f" % addBias)
        print("-"*15 + "> Apply " + self.costFunc.__name__ + " Cost Function")
        print(self.costFunc.__name__ +
              "(" + "%.5f" % addBias + ") = " + "%.5f" % self.costFunc(addBias))

# # w1 = 0, w2 = 1
# weights = np.array([0, 1])
# # b = 4
# bias = 4
# # Create Neuron
# n = Neuron(weights, bias, sigmoid)

# x1 = 2, x2 = 3
# x = np.array([2, 3])
# print(n.feedForward(x))

class NeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights, bias, sigmoid)
        self.h2 = Neuron(weights, bias, sigmoid)
        self.o1 = Neuron(weights, bias, sigmoid)
    
    def feedForward(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)
        out_o1 = self.o1.feedForward(np.array([out_h1, out_h2]))
        # Debug - Ini
        self.debug(x) if DebugNNtwrk else None
        # Debug - End
        return out_o1

    def debug(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)
        out_o1 = self.o1.feedForward(np.array([out_h1, out_h2]))
        
        Wcount = 0
        print("@"*15 +"> h1 - INI @" + "@"*15) if DebugNeuron else None
        self.h1.debug(x, Wcount) if DebugNeuron else None
        print("@"*15 + "> h1 val:" + "%.5f" % out_h1) if DebugNNtwrk else None
        print("@"*15 + "> h1 - END @" + "@"*15) if DebugNeuron else None
        print("*"*60) if DebugNeuron else None
        
        Wcount += len(self.h1.weights)
        print("@"*15 + "> h2 - INI @" + "@"*15) if DebugNeuron else None
        self.h2.debug(x, Wcount) if DebugNeuron else None
        print("@"*15 + "> h2 val:", "%.5f" % out_h2) if DebugNNtwrk else None
        print("@"*15 + "> h2 - END @" + "@"*15) if DebugNeuron else None
        print("*"*60) if DebugNeuron else None
        
        Wcount += len(self.h2.weights)
        print("@"*15 + "> o1 - INI @" + "@"*15) if DebugNeuron else None
        self.o1.debug(x, Wcount) if DebugNeuron else None
        print("@"*15 + "> o1 val:" + "%.5f" % out_o1) if DebugNNtwrk else None
        print("@"*15 + "> o1 - END @" + "@"*15) if DebugNeuron else None
        print("*"*60) if DebugNeuron else None

network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedForward(x))

# y_test = np.array([1,0,0,1])
# y_pred = np.array([0,0,0,0])
# print(mse_loss(y_test, y_pred))

