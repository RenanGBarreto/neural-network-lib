import numpy as np
import math

def randomizeArrayData(arr, variation=0.1):
    """ 
    Add a random variation on the values
    Param:
        - arr: The data
        - The variation to add
    Return: The data with the values changed
    """
    
    for i in range(len(arr)):
        arr[i] = arr[i] + (np.random.random()-0.5)*variation
        
    return arr

def createDataSetE1Q1(n, variation=0.1):
    """ 
    Creates a dataset: Classify the vertices of a cube
    Param:
        - n: The number of examples to generate
        - The variation to be added
    """
    
    X = []
    Y = []
    
    # Creates n/8 examples
    for i in range(math.ceil(n/8)):
        X.append(randomizeArrayData([0, 0, 0], variation)) 
        Y.append([0])
        X.append(randomizeArrayData([0, 0, 1], variation)) 
        Y.append([1])
        X.append(randomizeArrayData([0, 1, 0], variation)) 
        Y.append([2])
        X.append(randomizeArrayData([0, 1, 1], variation)) 
        Y.append([3])
        X.append(randomizeArrayData([1, 0, 0], variation)) 
        Y.append([4])
        X.append(randomizeArrayData([1, 0, 1], variation)) 
        Y.append([5])
        X.append(randomizeArrayData([1, 1, 0], variation)) 
        Y.append([6])
        X.append(randomizeArrayData([1, 1, 1], variation)) 
        Y.append([7])
    
    return X, Y