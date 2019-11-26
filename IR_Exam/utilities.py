# Define context manager to measure execution time.

import time
import pickle

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
    Saves a recommender system to file.
    """
    with open(filename, 'wb') as output:
        pickle.dump(rec, output, pickle.HIGHEST_PROTOCOL)
        
        
def loadRecSys(filename):
    """
    Loads a recommender system from file.
    """
    with open(filename, 'rb') as input:
        rec = pickle.load(input)
    return rec