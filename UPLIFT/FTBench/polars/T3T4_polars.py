import os
import sys
import polars as pl
import numpy as np
import warnings

from transformUtils import transform

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore') #cleaner, but not recommended

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def readNprep():
    print("Reading file: criteo_day21_10M")
    criteo = pl.scan_csv(os.path.join(SCRIPT_DIR, "../../datasets/criteo_day21_10M"), has_header=False)
    print(criteo.head().collect())

    cols = criteo.collect_schema().names()
    
    # note that 15/41 split is  defined by criteo_spec1.json/criteo_spec2.json (dummycode/recode ids 15–40 → 0-indexed 14–39).
    # also, "pt = [*range(0,14)]" is used in the sklearn implementation anyway.
    # polars does also have something like is_numeric, but it's determined sample-based, not whole-column-based, so less reliable.
    pt = cols[0:14]
    cat = cols[14:40]

    # mirrors fillna of sklearn implementation:
    # Replace NAs with 0 for numeric columns and empty string for categorical columns
    criteo = criteo.with_columns(
        [pl.col(c).fill_null(0) for c in pt]
        + [pl.col(c).fill_null("") for c in cat]
    )

    # Pandas infer the type of the first 14 columns as float/int.
    # SystemDS reads those as STRINGS and apply passthrough FT on those.
    # For a fair comparision, convert those here to str and later back to float
    criteo = criteo.with_columns(
        pl.col(pt).cast(pl.String)
    )

    return criteo

# Arguments 1 and 2 execute T3 and T4 respectively
specId = int(sys.argv[1])
if specId == 1:
    specfile = "criteo_spec1.json"
    resfile = "criteo10M_s1_pl.dat"
if specId == 2:
    specfile = "criteo_spec2.json"
    resfile = "criteo10M_s2_pl.dat"
print(resfile)

X = readNprep()

if specId == 1:
    X_prep = transform(X, specfile, resfile)

    times = np.loadtxt(os.path.join(SCRIPT_DIR, resfile))
    avgTime = round(times.mean() / 1000, 1) #sec
    filename = "Tab3_T3_pl.dat"
    with open(filename, "w") as file:
        file.write(str(avgTime))
if specId == 2:
    X_prep = transform(X, specfile, resfile, scale=True)
