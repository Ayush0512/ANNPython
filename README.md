# ANNPython
Artificial Neural Network with multiple hidden layers in Python without external libraries.

FEATURES
•	ANN without using any external libraries implemented in Python language.
•	Easy to use on low-level machines.
•	Generalised according to the needs of user.
•	Can be improvised according to problem requirements with minimal knowledge of Python.
•	Uses multiple hidden layers to improve performance which can be specified at runtime.

LOGIC & TRAINING
An Artificial Neural Network (ANN) is an information processing paradigm that is inspired the brain. ANNs, like people, learn by example. An ANN is configured for a specific application, such as pattern recognition or data classification, through a learning process. Learning largely involves adjustments to the synaptic connections that exist between the neurons. In Computer Science, we model this process by creating “networks” on a computer using matrices. These networks can be understood as abstraction of neurons without all the biological complexities taken into account. 
The training of the neural network consists of the following steps:
1.	Forward Propagation:
•	Initially we define all the weights as random double values between 0 and 1.
•	Then for initial input layer, we take the input values provided in the training data and multiply those values by weights which we defined randomly. After, we take the summation of all the multiplied values and store it in a variable.
•	Y = ∑WiIi ; where Wi  and Ii  is the ith Weight and Input of the network.
•	Then, we pass the result i.e. Y value through an activation function to calculate the output for that neuron. The sigmoid function is preferred as activation function, since it normalises the result between 0 and 1.
•	Sigmoid Function = 1/(1+e-y), where y is the result calculated for the neuron layer.
•	Then, the calculated result is passed to the next layer and all the above steps are performed in iteration till we reach the final output layer.
•	After calculating the output for the final layer, we move to the next step.

2.	Back Propagation:
•	First, we calculate the error value i.e. the difference between the actual output and the expected output.
•	Error = 0.5*(Target – Output)2
•	Depending on the error, we adjust the weights by multiplying the error with the input and again with the gradient of the Sigmoid curve.
•	Wi += α*Error*Input*Output*(1-Output). Here, Output*(1-Output) is a derivative of sigmoid curve. Also, α is the learning rate defined for the model.
•	After updating the weights for the final layer, we move to the previous layers and perform the above steps.
Note: The above steps constitute for one Epoch or iteration. In order to get good results, the above process should be repeated for a few thousand epochs.

DOCUMENTATION
import random, math

random.seed(0)

def r_matrix(m, n, a = -0.5, b = 0.5):
    return [[random.uniform(a,b) for j in range(n)] for i in range(m)]

def sigmoid(x):
    return 1.0/ (1.0 + math.exp(-x))

def d_sigmoid(y):
    return y * (1.0 - y)

In the above code snippet, two libraries random and math are imported for later use in the code. Random function needs to be initialized by passing any integer value using seed() method. r_matrix function is defined which returns a matrix of order MxN which will be specified by the user containing random values between -0.5 and 0.5. After that, sigmoid function is defined which is used as activation function in the model and similarly d_sigmoid function i.e. derivative of sigmoid function.

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

In the above code snippet, we defined a class for the neural network “NN”. Under this, a constructor is defined for the class in which we pass the nodes for the input layer, hidden layers and output layer. In the variable “nO” and “nI”, the output and input nodes are stored. We take the last index value for the output node and the first index value for the input node. The rest remaining nodes we define it for the hidden layers. Since we don’t know the number of hidden layers beforehand the whole process will occur at program execution.
After these steps, functions for initializing neurons and weights are invoked.

def __initWeights(self):

        self.weights = [0.0] * self.nWeights

        for i in range(self.nWeights):
            n_in, n_out = self.wDims[i]
            self.weights[i] = r_matrix(n_in, n_out)


def __initNeurons(self):

        self.layers = [0.0] * self.nLayers

        for i in range(self.nLayers):
            self.layers[i] = [0.0] * self.dims[i]

In the above code snippet, two functions “initWeights” and “initNeurons” are used for initializing the weights and input neurons of all the layers present in the model. “initWeights” function makes use of the r_matrix function defined earlier. The “initNeurons” function initializes all the neurons for all layers in model to 0 initially.

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

In the above code, two functions “__activateLayer” and “__backProp” are defined. “__activateLayer” function calculates the output for each layer as defined earlier in forward propagation. It then normalizes the output by applying sigmoid activation and stores it in “layer[i][j]” matrix where, i and j are the connected layers. In “__backProp” function the layer number ‘i’ and delta are passed as parameters and using the error function defined earlier in Back Propagation, error value is calculated and added to the previous error value. After that, again delta is calculated and the new delta value is returned to the user.

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

In the above code, “__updateWeights” and “feedforward” functions are defined. “__updateWeights” function takes i as layer value, delta i.e. change in output and target value and alpha also known as learning rate as parameters. This functions updates the weights of the ith layer as defined earlier in Back Propagation. “feedforward” function first checks if the number of input values of the dataset provided is equal to the number of neurons present in the input layer. If that is true then the function will proceed else it will throw an exception and the program execution will halt. This function first updates the input layer neurons value from ‘x’ and then call “__activateLayer” function which will perform the feed-forward process as discussed earlier.

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

In the above function, “backPropLearn” function is implemented. It will take ‘y’ as a parameter which is the target of the dataset. First, the function will check if the dimension of the target is equal to the number of neurons present in the output layer. If it is true, then the program will proceed further else it will halt. Next, starting from the output layer, initial error is calculated which equates to the change in output and target. Using the error value, delta is calculated by a function of derivative of sigmoid multiplied by initial error value. This delta value is stored in delta_list and passed as an argument to “__updateWeights” functions which was defined earlier. Similarly, this process will takes place for all the layers in a backwards manner.

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

In the above code, two functions “predict” and “train” are defined. “train” function takes the training dataset as a parameter and trains the network accordingly. In the function, MAX value corresponds to the number of epochs. From the dataset, each tuple or record is extracted and two variables ‘x’ and ‘y’ are used to store the inputs and target value respectively. From the “train” function two functions are called, “feedforward” which takes ‘x’ as argument and “backpropagation” which takes ‘y’ as argument which will perform the necessary operations as discussed earlier. “predict” function is used for testing the model created for the given dataset. It takes input values of the testing dataset as argument and passes it to the “feedforward” function and returns the weights of the output layer in a list.

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


Next up is the “main” function. In this function, the input data for training is defined as t0-t4. The input data along with the output is stored in ‘T’ variable. “nn” is the object created of the “NN” class with the arguments as the neurons for all the layers. After that, “train” function is called with ‘T’ as an argument which will perform the training of the model. Afterwards, “predict” function is called and the output value along with the target value are displayed to the user.




INSTALLATION

This code can be executed on any python IDE using python 3.x.

CONTRIBUTE

•	Issue Tracker - https://github.com/Ayush0512/ANNPython/issues
•	Source Code - https://github.com/Ayush0512/ANNPython

SUPPORT

If you are having issues, please let me know.
I have a mailing list located at: aroraayush0512@gmail.com

