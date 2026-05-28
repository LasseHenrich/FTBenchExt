import polars as pl
import time
import numpy as np
import warnings

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore') #cleaner, but not recommended

def readNprep():
    adult = pl.read_csv("../../datasets/adult.data", decimal_comma=True, has_header=False)
    print(adult.head())
    
X = readNprep()