# [Lasse] Note: Couldn't test yet, keeps crashing WSL, probably out of memory

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
    kdd = pl.scan_csv(os.path.join(SCRIPT_DIR, "../../datasets/KDD98.csv"), has_header=False)
    print(kdd.head().collect())

    # mirrors iloc and drop of sklearn implementation:
    # isolate first 469 columns and drop header row which was read as a data row
    kddX = kdd.select(pl.nth(range(0, 469))).slice(1)

    # mirrors replace of sklearn implementation:
    # replace whitespace-only / empty strings with null
    kddX = kddX.with_columns([
        pl.when(pl.col(c).str.strip_chars() == "")
          .then(None)
          .otherwise(pl.col(c))
          .alias(c)
        for c in kddX.columns
    ])

    # mirrs fillna:
    # Replace NAs with before/after entries
    kddX = kddX.with_columns(
        pl.all().fill_null(strategy="forward").fill_null(strategy="backward")
    )
    
    # Cast categorical columns that have numbers to float first,
    # to avoid mix of int and float type strings (5, 5.0), which
    # increases # distinct values in a column
    st = [23, 24, *range(28, 42), 195, 196, 197, *range(362, 384), *range(412, 434)]
    kddX = kddX.with_columns([
        pl.col(kddX.columns[i]).str.strip_chars().cast(pl.Float64).cast(pl.String)
        for i in st
    ])

    print(kddX.describe())
    return kddX

X = readNprep()
X_prep = transform(X, "kdd_spec1.json", "kdd_pl.dat", scale=True)

times = np.loadtxt(os.path.join(SCRIPT_DIR, "kdd_pl.dat"))
avgTime = round(times.mean() / 1000, 1) #sec
filename = "Tab3_T2_pl.dat"
with open(filename, "w") as file:
    file.write(str(avgTime))
