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
#weights = [np.random.randn(y, x)for x, y in zip(sizes[:-1], sizes[1:])]
"""
for x, y in zip(sizes[:-1], sizes[1:]):
    self.weights.append(np.random.randn(y, x))
"""
def randomWB(layers):
    amount = len(layers)
    weight=[]
    bias=[]
    for x,y in (layers[:-1],layers[1:]):
        weight.append(np.random.randn(y,x))
    for y in layers[1:]:
        bias.append(np.random.randn(y,1))
    return weight,bias
def slicer(data,sizeFraction):
    sizeData =len(data)
    batch=[]
    for i in range(0,sizeData,sizeFraction):
        batch.append(data[i:i+sizeFraction])
    return batch

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
    def cost(self,outputActivation,y):
        """
        partial derivative of cost function. Returns vector for each run.               
        """
        sub=outputActivation-y
        return sub



    #epochs: number of times training vectors are used. 
    #rate: learning rate of each iteration. 
    #mini_batch_size: size of each of the batches of data
    def descentApplication(self,data, epochs,mini_batch_size,rate):
        train=list(data)
        size=len(train)
        for i in range(epochs):
            shuffle(train)
            mini_batches=slicer(train,mini_batch_size)
            for element in mini_batches:
                self.descent(element,rate)
    
    def descent(self,mini_batch,rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        size_batch = len(mini_batch)
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backpropagation(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weight = [w-(rate/size_batch)*nw for w,nw in zip(self.weight,nabla_w)]
        self.bias = [b-(rate/size_batch)*nb for b,nb in zip(self.bias,nabla_b)]
    def backpropagation(self,x,y):
        nablaBias,nablaWeight=[],[]
        for i in self.bias:
            nablaBias=np.append(nablaBias,np.zeros(i.shape))
        for i in self.weight:
            nablaWeight=np.append(nablaWeight,np.zeros(i.shape))
        activation=[x]
        z=[]
        for w,b in zip(self.weight,self.bias):
            zValue=(w @ activation) + b
            z.append(zValue)
            activations=sigmoid(zValue)
            activation.append(activations)
        last=activation[-1]
        delta=cost()

        return nablaB,nablaW

        











