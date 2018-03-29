import numpy as np
import math

def createStepArray(arraySize, stepSize):
    """ 
    Creates a new array of numbers spaced by stepSize
    """
    arr = []
    
    for i in range(arraySize):
        arr.append(stepSize*i)
        
    return arr


def createDataSetL1Q5(datasetSize, stepSize, lookahead=1):
    """ 
    Generates dataset for training using Q5 function and the fact that each entry depends on the anterior one
    """
    
    X = range(0, datasetSize, stepSize)
    Y = []
    
    for xi in X:
        Y.append(np.sin(xi + (np.square(np.sin(X[i-1]))))) 
    
    # Creates datasetSize examples
    for i in range(1, datasetSize):
        Y.append(np.sin(X[i-1] + (np.square(np.sin(X[i-1]))))) # Desired function, in which each entry depends on the anterior
        
    return X, Y

display(generateDataset_Q5(20, 0.1))