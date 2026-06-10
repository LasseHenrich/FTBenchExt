import os
import polars as pl
import time
import numpy as np
import warnings

from transformUtils import transform

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore') #cleaner, but not recommended

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def readNprep():
    adult = pl.scan_csv(os.path.join(SCRIPT_DIR, "../../datasets/adult.data"), decimal_comma=True, has_header=False)
    print(adult.head())
    
    # Pandas infer the type of a few columns as int64.
    # SystemDS reads those as STRINGS and apply passthrough FT on those.
    # For a fair comparision, convert those here to str and later back to float
    adult = adult.with_columns(
        pl.nth(range(0,15)).cast(pl.String)
    )
    #print(adult.info())
    return adult
    
X = readNprep()
X_prep = transform(X, "adult_spec2.json", "adult_pl.dat")