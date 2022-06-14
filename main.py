import sys
import numpy as np
from config import config, create_folder
from benchmarkMixed import benchmark

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

"""
Main entry of the program
Creates the logging files, loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""

def main():
    # Setting up logging
    create_folder()

    # For being able to see progress that some methods use verbose (for debugging purposes)
    f = open(config['model_dir'] + '/console.out', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)


    #Load the data
    with np.load('data\LR_task_with_antisaccade_synchronised_min_hilbert.npz') as f:
        trainX_PA = f[config['trainX_variable']]
        trainY_PA = f[config['trainY_variable']]


    with np.load('data\LR_task_with_dots_synchronised_min_hilbert.npz') as f:
        trainX_LG = f[config['trainX_variable']]
        trainY_LG = f[config['trainY_variable']]


    # The LG is shorter in size, so we cut down the PA dataset
    shorterLength = len(trainY_LG) 
    trainX_PA, trainY_PA = trainX_PA[:shorterLength], trainY_PA[:shorterLength]
    
    #Start benchmark
    benchmark(trainX_PA, trainY_PA, trainX_LG, trainY_LG)

if __name__=='__main__':
    main()
