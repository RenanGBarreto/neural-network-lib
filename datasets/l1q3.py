import numpy as np
import math, random

def randomizeArrayDataL1Q3A(arraySize):
    """ 
    Creates a new array of boolean pairs in int format
    Param:
        - arraySize: The array size
    Return: array of boolean pairs
    """

    arr = []
    
    for i in range(arraySize):
        arr.append([(np.random.randint(0,2)),(np.random.randint(0,2))])
        
    return arr

def createDataSetL1Q3A(datasetSize):
    """ 
    Generates dataset for XOR function training for L1Q3A
    Param:
        - datasetSize: The number of examples to generate
    Return: Two arrays with the dataset X and Y
    """
    
    X = randomizeArrayDataL1Q3A(datasetSize)
    Y = []
    
    # Creates datasetSize examples
    for i in range(datasetSize):
        if X[i] == [0, 0]: 
            Y.append([0])
        if X[i] == [0, 1]: 
            Y.append([1])
        if X[i] == [1, 0]: 
            Y.append([1])
        if X[i] == [1, 1]: 
            Y.append([0])
        
    return X, Y

def randomizeArrayDataL1Q3B(arraySize):
    """ 
    Creates a new array of random float numbers between 0 and 4
    Param:
        - arraySize: The array size
    Return: array of random float numbers between 0 and 4
    """
    arr = []
    
    for i in range(arraySize):
        arr.append([np.random.uniform(0.00001, 4.0)]) 
        
    return arr

def createDataSetL1Q3B(datasetSize):
    """ 
    Generates dataset for training following L1Q3B function format
    Param:
        - datasetSize: The number of examples to generate
    Return: Two arrays with the dataset X and Y
    """
    
    X = randomizeArrayDataL1Q3B(datasetSize)
    Y = []
    
    # Creates datasetSize examples
    for i in range(datasetSize):
        Y.append([(np.sin(np.pi * X[i][0])) / (np.pi * X[i][0])])
        
    return X, Y
