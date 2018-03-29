import numpy as np
import math

def createDataSetL1Q5(n, lookback=5, lookahead=1):
    """ 
    Generates dataset for training using Q5 function
    Param:
        - n: THe number of points to extract
        - lookback: how many points we will see on the past
        - lookahead: how many points in future we will try to predict
    Return: The serie
    """
    
    assert lookahead >= 1
    assert lookback >= 0
    x_points = np.linspace(0, 50, n)
    
    serie = []
    
    for xi in x_points:
        serie.append( np.sin(xi + (np.square(np.sin(xi)))) ) 

    X = []
    y = []
    for i in range(lookback, len(x_points)-lookahead):
        
        x_data = []
        for c in range(lookback, 0, -1):
            x_data.append(serie[i-c])
        X.append(x_data)
        
        y.append([serie[i+lookahead]])
    
    return X, y