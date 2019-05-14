import numpy as np
import math
from random import shuffle 

allFiles = np.load('./data/all_data.npy')

#reference: http://neuralnetworksanddeeplearning.com/chap1.html#the_architecture_of_neural_networks
# represents the neural network to be used. it fills the biases and weights vector with random values
# biases is an array of dimensions of the second layer.
# weights are filled from neuron1 to neuron2 and neuron2 to neuron3 and so on and so forth
# batch sizes
# learning Rate: speed of learning
# Lamb: Lambda is the vector to eliminate varianza
def randomWB(layers):
    amount = len(layers)
    weight = np.zeros([])
    bias = np.array([])
    for i in range(amount-1):
        first = layers[i]
        second = layers[i + 1]
        print("first",first)
        print("second",second)
        weight=np.append(weight, np.zeros([second, first]))
        np.append(bias,np.zeros(([layers[i+1]])))
    return weight,bias
def sigmoid(z):
    value = 1/(1+np.exp(-z))
    return value
class Network(object):
    def __init__(self,layers):
    #layers: list of layers
    #numLayers: number of layers. Entrance/mystery/output
    #weight: matrix where rows are # of neurons in the next layer. Column is number of nueron in current layer. 
    #bias: value that guides to right answer
        self.layers=layers
        self.numLayers = len(layers)
        self.weight,self.bias = randomWB(layers)
    #a: column vector with the activation values of the present layer. 
    #feedforward: a = sigmoide(wa+b)
    def feedforward(self,a):
        for b in self.bias:
            for w in self.weight:
                a = sigmoid((b @ w)+a)
        return a 
    def cost(self, self.weight,self.bias):


    #epochs: number of times training vectors are used. 
    #rate: learning rate of each iteration. 
    #mini_batch_size: size of each of the batches of data
    def gradDescent(self,data, epochs,mini_batch_size,rate):
        train=list(data)
        size=len(train)
        for i in range(epochs):
            shuffle(train)
            mini_batches=[train[k:k + mini_batch_size] for j in range(0,size,mini_batch_size)]
            for element in mini_batches:
                self.setBatch(element,rate)
    
    def setBatch(sel,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weight]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backpropagation(x,y)
            nabla_b = []
    
    def backpropagation(self,x,y):









