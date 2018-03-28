import numpy as np

def unisonShuffledCopies(a, b):
    """
    Shuffle two diffrent arrays but keep the link between them.
    Param:
        - a: The first array
        - b: The second array
    Return: A random permutation of the arrays (linked data)
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]  
