# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:06:17 2019

@author: Ayush
"""

import random, math

random.seed(0)

def r_matrix(m, n, a = -0.5, b = 0.5):
    return [[random.uniform(a,b) for j in range(n)] for i in range(m)]

def sigmoid(x):
    return 1.0/ (1.0 + math.exp(-x))

def d_sigmoid(y):
    return y * (1.0 - y)
class NN:

    def __init__(self, dims):

        self.dims    = dims
        self.nO      = self.dims[-1]
        self.nI      = self.dims[0]
        self.nLayers = len(self.dims)
        self.wDims   = [ (self.dims[i-1], self.dims[i])\
                         for i in range(1, self.nLayers) ]

        self.nWeights = len(self.wDims)

        self.__initNeurons()
        self.__initWeights()

    def __initWeights(self):

        self.weights = [0.0] * self.nWeights

        for i in range(self.nWeights):
            n_in, n_out = self.wDims[i]
            self.weights[i] = r_matrix(n_in, n_out)


    def __initNeurons(self):

        self.layers = [0.0] * self.nLayers

        for i in range(self.nLayers):
            self.layers[i] = [0.0] * self.dims[i]
    def __activateLayer(self, i):

        prev = self.layers[i-1]
        n_in, n_out = self.dims[i-1], self.dims[i]

        for j in range(n_out):
            total = 0.0
            for k in range(n_in):
                total += prev[k] * self.weights[i-1][k][j]   # num weights is always one less than num layers

            self.layers[i][j] = sigmoid(total)


    def __backProp(self, i, delta):

        n_out, n_in = self.dims[i], self.dims[i+1]
        next_delta  = [0.0] * n_out

        for j in range(n_out):
            error = 0.0
            for k in range(n_in):
                error += delta[k] * self.weights[i][j][k]
            pred = self.layers[i][j]
            next_delta[j] = d_sigmoid(pred) * error

        return next_delta


    def __updateWeights(self, i, delta, alpha = .7):

        n_in, n_out = self.wDims[i]

        for j in range(n_in):
            for k in range(n_out):

                change = delta[k] * self.layers[i][j]
                self.weights[i][j][k] += alpha * change


    def feedForward(self, x):
        if len(x) != self.nI:
            raise ValueError('length of x must be same as num input units')


        for i in range(self.nI):
            self.layers[0][i] = x[i]

        for i in range(1, self.nLayers):
            self.__activateLayer(i)

    def backPropLearn(self, y):
        if len(y) != self.nO:
            raise ValueError('length of y must be same as num output units')


        delta_list = []
        delta      = [0.0] * self.nO

        for k in range(self.nO):
            pred  = self.layers[-1][k]
            error = y[k] - pred

            delta[k] = d_sigmoid(pred) * error


        delta_list.append(delta)
        for i in reversed(range(1, self.nLayers-1)):
            next_delta = self.__backProp(i, delta)

            delta = next_delta
            delta_list = [delta] + delta_list

        # now perform the update
        for i in range(self.nWeights):
            self.__updateWeights(i, delta_list[i])
            
    def predict(self, x):
        self.feedForward(x)

        return self.layers[-1]

    def train(self, T):

        i, MAX = 0, 5000

        while i < MAX:
            for t in T:
                x, y = t
                self.feedForward(x)
                self.backPropLearn(y)

            i += 1
            
def main():
    # no. 0
    t0 = [ 0,1,1,1,0,\
           0,1,0,1,0,\
           0,1,0,1,0,\
           0,1,0,1,0,\
           0,1,1,1,0 ]
    
    # no. 1
    t1 = [ 0,1,1,0,0,\
           0,0,1,0,0,\
           0,0,1,0,0,\
           0,0,1,0,0,\
           1,1,1,1,1 ]
    
    # no. 2
    t2 = [ 0,1,1,1,0,\
           0,0,0,1,0,\
           0,1,1,1,0,\
           0,1,0,0,0,\
           0,1,1,1,0 ]
    
    # no. 3
    t3 = [ 0,1,1,1,0,\
           0,0,0,1,0,\
           0,1,1,1,0,\
           0,0,0,1,0,\
           0,1,1,1,0 ]
    
    # no. 4
    t4 = [ 0,1,0,1,0,\
           0,1,0,1,0,\
           0,1,1,1,0,\
           0,0,0,1,0,\
           0,0,0,1,0 ]
                
    
    
    
    T = [(t0, [1,0,0,0,0]),(t1, [0,1,0,0,0]), (t2, [0,0,1,0,0]), (t3, [0,0,0,1,0]), (t4, [0,0,0,0,1])]
    
    nn = NN([25, 50, 50, 5])
    nn.train(T)
    t5 = [ 0,1,1,1,0,\
           0,0,0,1,0,\
           0,1,1,1,0,\
           0,0,0,1,0,\
           0,1,1,1,0 ]
    o5 = [0,0,0,1,0]
    
    x = nn.predict(t5)
    print("The Target Matrix is: ",o5)
    print("The Output Matrix is: ",x)
    
if __name__=='__main__':
    main()