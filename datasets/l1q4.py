import numpy as np
import math, random

def randomizeArrayDataL1Q4(arraySize):
    """ 
    Creates a new array of float pairs (vectors) between -1 and 1, corresponding to X and Y coordinates
    of the points inside the circle
    Param:
        - arraySize: The array size
    Return: a new array of float pairs
    """
    arr = []
    
    for i in range(arraySize):
        arr.append([(np.random.uniform(-1.0, 1.0)), (np.random.uniform(-1.0, 1.0))])
    
    return arr

def checkPointLine(x1, y1, x2, y2, xp, yp):
    """ 
    Checks if a point (xp, yp) is above or below a line defined by other tho points (x1,y1) and (x2,y2)
    """
    v1 = [x2-x1, y2-y1]   # Vector 1
    v2 = [xp-x1, yp-y1]   # Vector 2
    crossProduct = v1[0]*v2[1] - v1[1]*v2[0]
    
    # DEBUG
    #if crossProduct > 0:
        #print 'point is on the counter-clockwise side of line'
    #elif crossProduct < 0:
        #print 'point is on the clockwise side of line'
    #else:
        #print 'point is exactly on the line'
        
    return crossProduct

def createDataSetL1Q4(datasetSize):
    """ 
    Generates dataset for XOR function training
    """
    
    X = randomizeArrayDataL1Q4(datasetSize)
    
    # X below is for debugging reasons
    #X = [[0.3, 0.3], [0.9, 0.9], [0.3, -0.3], [0.9, -0.9], [-0.3, 0.3], [-0.9, 0.9], [-0.3, -0.3], [-0.9, -0.9]]
    
    Y = []
    
    # Creates datasetSize examples
    for i in range(datasetSize):
        if X[i][0] >= 0:
            
            if X[i][1] >= 0:
                
                if checkPointLine(0.0, 1.0, 1.0, 0.0, X[i][0], X[i][1]) >= 0: # Points for the line on the upper right quadrant
                    Y.append([5]) #Point is above line, so its on C5 region
                    
                else:
                    Y.append([1]) #Point is below line, so its on C1 region
                    
                #print('Superior right')
            else:
                
                if checkPointLine(0.0, -1.0, 1.0, 0.0, X[i][0], X[i][1]) <= 0: # Points for the line on the lower right quadrant
                    Y.append([8]) #Point is below line, so its on C8 region
                    
                else:
                    Y.append([4]) #Point is above line, so its on C4 region
                    
                #print('Inferior right')
            
        else:
            if X[i][1] >= 0:
                
                if checkPointLine(0.0, 1.0, -1.0, 0.0, X[i][0], X[i][1]) <= 0: # Points for the line on the upper left quadrant
                    Y.append([6]) #Point is above line, so its on C6 region
                    
                else:
                    Y.append([2]) #Point is below line, so its on C2 region
                    
                #print('Superior left')
                
            else:
                
                if checkPointLine(0.0, -1.0, -1.0, 0.0, X[i][0], X[i][1]) >= 0: # Points for the line on the lower left quadrant
                    Y.append([7]) #Point is below line, so its on C7 region
                    
                else:
                    Y.append([3]) #Point is above line, so its on C3 region
                
                #print('Inferior left')
                
    return X, Y
