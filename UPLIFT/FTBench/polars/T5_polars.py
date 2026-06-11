import os
import polars as pl
import numpy as np
import warnings

from transformUtils import transform

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore') #cleaner, but not recommended

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def readNprep():
    train = pl.scan_csv(os.path.join(SCRIPT_DIR, "../../datasets/santander.csv"), has_header=False)

    # mirrors iloc
    cols = train.collect_schema().names()
    santander = train.select(pl.nth(range(2, len(cols)))).slice(1)

    # mirrors "columns = ...", i.e. column renaming by constructing a mapping
    santander = santander.rename({old: str(i) for i, old in enumerate(cols[2:])}) #rename header from 0 to 199
    print(santander.head().collect())

    return santander

X = readNprep()
X_prep = transform(X, "santander_spec2.json", "santander_pl.dat")
