# Define context manager to measure execution time.

import time
import numpy as np

class codeTimer:
    """
    Context manager, measures and prints the execution time of a function.
    """
    
    def __init__(self, name=None):
        self.name = "Executed '"  + name + "'. " if name else ""

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = (self.end - self.start)
        print('%s Elapsed time: %0.6fs' % (str(self.name), self.elapsed))

        
def saveRecSys(rec, filename):
    """
    Saves the user and item matrices of a recommender system to file.
    """
    
    np.savez(filename, X = rec.X, Y = rec.Y)
        
        
def loadRecSys(rec, filename):
    """
    Loads the user and item matrices of a recommender system from file.
    """
    
    data = np.load(filename)
    rec.X = data['X']
    rec.Y = data['Y']