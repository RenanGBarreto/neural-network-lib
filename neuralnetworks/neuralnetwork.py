import numpy as np
import math
import pprint
import pickle
from viznet import connecta2a, connect121, node_sequence, NodeBrush, EdgeBrush, DynamicShow
from sklearn.preprocessing import OneHotEncoder
import pdb
pdb.set_trace = lambda: 1  # This solves a problem with the python debugger and the library viznet

class Neuron:
    """
    Class that represents a neuron inside the neural network
    """

    def __init__(self, inputs, activation='sigmoid'):   
        """
        Create a neuron object.
        Param:
            - inputs: THe number of inputs that this neuron will have.
            - activation: The name of the activation function
        """

        self.weights = np.random.rand(inputs)
        self.bias = np.random.random() + 0.001 # Just to make sure it is not zero :)
        
        self.newWeights = self.weights
        self.newBias = self.bias
        
        assert inputs > 0
        self.inputs = int(inputs)
        
        assert activation == 'sigmoid' or activation == 'tanh' or activation == 'relu' or activation == 'linear'
        self.activation = activation

    def activation_function(self, func_type, value):
        """
        Most commons activation functions.
        Param: func_type - A tring with one of the valid values: sigmoid, tanh, relu, linear
                    value - Apply the function to the value
        Return: The result of the activation function
        """
        
        assert func_type == 'sigmoid' or func_type == 'tanh' or func_type == 'relu' or func_type == 'linear'
        
        if func_type == 'sigmoid':
            return 1 / (1 + np.exp(-value))
        elif func_type == 'tanh':
            return math.sinh(value) / math.cosh(value)
        elif func_type == 'relu':
            return 0 if (value < 1) else value
        elif func_type == 'linear':
            return value    
    
    def adjustWeights(self):
        """
        Update the weigths with the pre-calculated newWeigths variable. This function is used during training.
        """
        self.weights = self.newWeights
        self.newWeights = self.weights
        
        self.bias = self.newBias
        self.newBias = self.bias
        
    
    def foward(self, x, apply_activation=True, verbose=False):
        """
        Calculate the output of the neuron. 
        Param:
            - inputs: Array with the inputs signals
            - apply_activation: If true, Apply the activation final. If false, return the output without activation function (Vj)
            - verbose: If true, show some data on the stdout
        Return: The neuron output (with or without the activation function applied)
        """
        vj = np.dot(x, self.weights) + self.bias
        out = vj if (not apply_activation) else self.activation_function(self.activation , vj)
        
        if verbose:
            print("Input: ", x, " -> Vj: ", vj, " -> Output: ", out)
            
        return out
        
    def __repr__(self):
        """
        Alternative way to see the object as string
        """
        return pprint.pformat(vars(self), indent=0, width=1000, depth=None)



class NeuralNetwork:
    """
    Class that represents a Feed-foward Neural Network that can be trained using backpropagation.
    """
    
    def __init__(self, inputs=1, architecture=[1], lr=0.01, momentum=0, isClassification=False, autoEncode=False, activation='sigmoid', activation_last_layer='linear', seed=42):
        """
        Create a feed-foward neural network object.
        Param:
            - inputs: The amount of inputs the network will have (scalar)
            - architecture: An array of numbers that represents the amount of neurons in each layer            
            - lr: The learning rate
            - momentum: The momentum constant. Zero if we shall not momentum
            - isClassification: If true, use a int+round function that is applyed in the end of the calculation
            - autoEncode: Auto encode the output class
            - activation: The name of the activation function for all hidden and input layers
            - activation_last_layer: Activation function for the last layer of the model
            - seed: The random seed
        """
        
        # Make sure the architecture is a list and not empty
        assert isinstance(architecture, list) and len(architecture) > 0
        
        np.random.seed(seed)
        
        self.architecture = architecture        
        assert inputs > 0
        self.inputs = int(inputs)
        self.momentum = momentum if momentum > 0 else 0
        self.lr = lr if lr > 0 else 0.00001 # Make sure the LR is not too close to zero       
        self.isClassification = isClassification        
        self.activation = activation
        self.activation_last_layer = activation_last_layer
       
        # This helps with classification tasks
        self.autoEncode = autoEncode
        if self.autoEncode:
            self.encoder = OneHotEncoder(sparse=False)

        # Create the layers and neurons based on the architecture
        self.initLayers()
        
    def initLayers(self):
        """
        Initialize the layers and create the neurons
        """
        
        # The amount of inputs of each Neuron need to be the total of neurons of the previus layer OR self.inputs if this is the first layer
        totalInputs = self.inputs
        
        # Clear all layers (and weigths)
        self.layers = []
        
        # For each layer of the architecture
        for lIdx in range(len(self.architecture)):  
            
            # Start an array with the current layer
            currentLayer = []
            
            # The last layer can have a different activation function
            isLastLayer = (lIdx == len(self.architecture)-1)
            
            # Create n neurons 
            for count in range(self.architecture[lIdx]):
                
                currentLayer.append( Neuron(totalInputs, activation=(self.activation_last_layer if isLastLayer else self.activation)))
                
            totalInputs = self.architecture[lIdx]
            
            # Add the current layer to the layer list
            self.layers.append(currentLayer)        
     
    def activation_function_derivative(self, func_type, value):
        """
        Most commons activation functions.
        Param: func_type - A string with one of the valid values: sigmoid, tanh, relu, linear and step
                    value - Apply the derivative function to the value
        Return: The result of the derivative activation function
        """
        
        assert func_type == 'sigmoid' or func_type == 'tanh' or func_type == 'relu' or func_type == 'linear'
        
        if func_type == 'sigmoid':
            f = 1/(1+np.exp(-value))
            df = f * (1 - f)
            return df
        elif func_type == 'tanh':
            f = math.sinh(value) / math.cosh(value)
            df = (1-(f*f))/2
            return df
        elif func_type == 'relu':
            return 0 if (value < 0) else 1
        elif func_type == 'linear':
            return 1

            
    
    def predict(self, X, verbose=False):
        """
        Create the predictions for the input samples.
        Param:
            X: The dataset of examples
            verbose: If true, show some data on the stdout
        Return: The predicted values
        """

        y = []
        
        for currentExample in X: 
        
            # On the first layer, the input is the data row
            currentInputs = currentExample

            currentOutputs = []

            # For each layer
            for l in range(len(self.layers)):
                currentOutputs = []

                currentLayer = self.layers[l]

                # For each neuron 
                for n in range(len(self.layers[l])):
                    currentNeuron = self.layers[l][n]                    
                    currentOutputs.append(currentNeuron.foward(currentInputs, apply_activation=True, verbose=verbose))

                # For the next layer, the input become the output of the preview layer
                currentInputs = currentOutputs

            if self.isClassification:
                if self.autoEncode: 
                    # If it is an autoEncode, do something similar to a "softmax"
                    biggestIdx = 0
                    biggestValue = -999999
                    for idx in range(len(currentOutputs)):
                        if currentOutputs[idx] >= biggestValue:
                            biggestValue = currentOutputs[idx]
                            biggestIdx = idx

                    for idx in range(len(currentOutputs)):
                        currentOutputs[idx] = 1 if idx == biggestIdx else 0
                else:
                    # Apply the output filter if needed
                    for i in range(len(currentOutputs)):
                        currentOutputs[i] = currentOutputs[i] if not self.isClassification else int(round(currentOutputs[i])) 

            # if autoencode is false, lets decode the class
            if self.autoEncode:
                y.append(np.dot([currentOutputs], self.encoder.active_features_).astype(int))
            else:
                y.append(currentOutputs)
            
        return y
    
    def save(self, file='neuralnet.pkl'):
        """
        Serialize the Neural Network to a file that can be loaded later.
        Param:
            file: The file path where the object will be saved
        """
        assert file != None
        with open(file, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
    def load(self, file='neuralnet.pkl'):
        """
        Serialize the Neural Network to a file that can be loaded later.
        Param:
            file: The file path where the object will be saved
        Return: This object with the new architecture
        """
        assert file != None
        with open(file, 'rb') as input:
            tmp = pickle.load(input)
        self.__dict__.update(tmp.__dict__)
        
        return self
        
    def draw(self, file=None, size=(10,6)):
        """
        Draw the network architecture 
        Param:
            - file: The file where the image will be saved. Default: None
            - size: the image size. Default: (10,6)
        """
        
        with DynamicShow(size, filename=file) as d:
            
            num_hidden_layer = len(self.architecture) - 2
            token_list = ['\sigma^z'] + ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
            kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
            radius_list = [0.1] + [0.2] * num_hidden_layer + [0.3]
            x_list = 1.5 * np.arange(len(self.architecture)) + 1.5

            seq_list = []
            
            # Input pins
            inputPins = NodeBrush('qc.C', d.ax)
            seq_list.append(node_sequence(inputPins, self.inputs, center=(0, 0), space=(0, 1)))
            
            # Network and connections
            for n, kind, radius, y in zip(self.architecture, kind_list, radius_list, x_list):
                b = NodeBrush(kind, d.ax)
                seq_list.append(node_sequence(b, n, center=(y, 0), space=(0, 1)))
            
            for st, et in zip(seq_list[:-1], seq_list[1:]):
                connecta2a(st, et, EdgeBrush('-->', d.ax))
            
            # Output pins
            outputEdge = EdgeBrush('---', d.ax)
            outputPins = NodeBrush('qc.C', d.ax)
            seq_list.append(node_sequence(outputPins, self.architecture[-1], center=(x_list[-1]+1.5, 0), space=(0, 1)))            
            connect121( seq_list[-2], seq_list[-1], outputEdge)                
                    
    def __repr__(self):
        """
        Alternative way to see the object as string
        """
        return pprint.pformat(vars(self), indent=1, width=1, depth=5)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=True, validation_split=0.1, shuffle=True):
        """
        Train the Neural Network
        Param:
            x: The input samples data, as a Numpy array (or list of Numpy arrays).
        Return: Numpy array(s) with the training statistics
        """       
        
        # if we are on a classification tasks, lets automaticaly encode the classes
        if self.autoEncode:
            self.encoder.fit(y)
        
        X_train = x        
        y_train = y if not self.autoEncode else self.encoder.transform(y)
        
        ## TODO shuffle and split the validation dataset
        X_val = x 
        y_val = y if not self.autoEncode else self.encoder.transform(y)
 
        assert len(y_train[0]) == self.architecture[-1]
        assert len(y_val[0]) == self.architecture[-1]

        # Discover the corect bash size
        if batch_size == None or batch_size > len(X_train):
            batch_size = len(X_train)
            
        # For each epoch
        for e in range(epochs):
            
            # TODO shuffle
            
            # For each example
            for idx in range(len(X_train)):
                
                # Foward pass
            
                # For each layer

                outputs = []
                vjs = []
                
                currentInputs = X_train[idx]
                currentOutputs = []
                currentVjs = []

                for l in range(len(self.layers)):
                    currentOutputs = []
                    currentVjs = []

                    currentLayer = self.layers[l]

                    # For each neuron 
                    for n in range(len(self.layers[l])):
                        currentNeuron = self.layers[l][n]                    
                        currentOutputs.append(currentNeuron.foward(currentInputs, apply_activation=True, verbose=verbose))                        
                        currentVjs.append(currentNeuron.foward(currentInputs, apply_activation=False, verbose=False))

                    # For the next layer, the input become the output of the preview layer
                    currentInputs = currentOutputs
                    outputs.append(currentOutputs)
                    vjs.append(currentVjs)   
                                    
                # Backwards pass
                
                lastLayerGradients = []
                
                # For each layer (reverse order)
                for l in range(len(self.layers)-1, -1, -1):
                    
                    currentLayerGradients = []
                    
                    # For each neuron
                    for n in range(len(self.layers[l])):
                        
                        currentNeuron = self.layers[l][n] 
                        
                        # Error for the last layer
                        if l == len(self.layers) - 1:                            

                            derivative = self.activation_function_derivative(currentNeuron.activation, vjs[l][n])
                            grad = (outputs[l][n] - y_train[idx][n]) * derivative                  
                                                        
                        else:
                            
                            # Lets try to calculated the weighted sum of the gradients and weights 
                            wSum = 0
                            for nextNeuronIdx in range(len(self.layers[l+1])):                                                               
                                wkj = self.layers[l+1][nextNeuronIdx].weights[n]
                                gradj = lastLayerGradients[nextNeuronIdx]                                
                                wSum += wkj*gradj
                            
                            wSum += self.layers[l][n].bias # add the bias
                            
                            vjDerivative = self.activation_function_derivative(currentNeuron.activation, vjs[l][n])                            
                            grad = wSum * vjDerivative
                             
                        currentLayerGradients.append(grad)
                        
                        for w in range(len(currentNeuron.weights)):

                            if l == 0:
                                delta = self.lr * grad * X_train[idx][w]
                            else:
                                delta = self.lr * grad * outputs[l-1][w]

                            currentNeuron.newWeights[w] = currentNeuron.weights[w] - delta
                        
                        # Lets try to adjust the bias too
                        currentNeuron.newBias = currentNeuron.bias - delta * abs(currentNeuron.bias)
                                                    
                    lastLayerGradients = currentLayerGradients

                # Lets update all weights of the network at once
                for l in range(len(self.layers)):                    
                    # For each neuron 
                    for n in range(len(self.layers[l])):
                        self.layers[l][n].adjustWeights()
        
        return None