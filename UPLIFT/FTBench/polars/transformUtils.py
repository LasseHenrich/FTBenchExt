import time

import polars as pl
import numpy as np
import json


def getTransformSpec(X, filename):
    # Deserialize the transform spec
    fullfile = "../systemds/specs/" + filename
    specFile = open(fullfile, "r")
    spec = json.loads(specFile.read())

    # Read column list for individual encoders
    rc = spec.get('recode')
    dc = spec.get('dummycode')
    fh = spec.get('hash')
    bins = []
    methods = []
    numbins = []
    # TODO: support column specific methods and numbins
    equiWidth = True
    binCount = 0;
    if (spec.get('bin')):
        for bin in spec['bin']:
            bins.append(bin.get('id'))
            methods.append(bin.get('method'))
            numbins.append(bin.get('numbins'))
        equiWidth = True if methods[0] == "equi-width" else False
        binCount = numbins[0]

    bins = [i-1 for i in bins]
    # Union the columns lists for categorical encoders
    # NOTE: DC list may contain binned columns 
    catEncode = []
    if fh:
        fh = [i-1 for i in fh]
        catEncode = catEncode + fh
    if rc:
        rc = [i-1 for i in rc]   #convert to 0-based indices
        catEncode = catEncode + rc
    if dc:
        dc = [i-1 for i in dc]
        catEncode = catEncode + dc
    # Sort and remove duplicates
    catEncode = list(set(sorted(catEncode)))
    # Get categorical column list by subtracting binned from catEncode
    catCols = list(set(catEncode) - set(bins))
    # Passthrough list = all columns - (numeric + categorical)
    pt = list(set(range(0,len(X.columns))) - set(catCols + bins))
    return {'rc':rc, 'dc':dc, 'fh':fh, 'bins':bins, 'method':equiWidth, 'numbin':binCount, 'cat':catCols, 'pt':pt}

def buildMathExpressions(X: pl.LazyFrame, encoders: dict):
    expr_list = []
    
    # Passthrough (convert to float)
    if 'pt' in encoders:
        for idx in encoders['pt']:
            col_name = X.columns[idx]
            # strip_chars as the numbers are e.g. " 9", " 13", etc.
            expr = pl.col(col_name).str.strip_chars().cast(pl.Float64)
            expr_list.append(expr)
    
    # TODO: scale if transform was called with scale=True
    
    # Binning
    if 'bins' in encoders:
        nbins = encoders['numbin']
        print(nbins)
        for idx in encoders['bins']:
            col_name = X.columns[idx]
            # TODO: the cast in scikit-learn/transformUtils.py (line 80) currently happens BEFORE starting the timers
            expr = pl.col(col_name).str.strip_chars().cast(pl.Float64)
            if encoders['method']: # uniform/equi-width
                # TODO: each bin must have the same width, regardless of number of elements
                expr = expr
            else: # quantile
                # equal number of elements per bin
                expr = expr.qcut(nbins)
            expr_list.append(expr)
            
    # Recoding (ordinal encoding)
    if 'rc' in encoders:
        for idx in encoders['rc']:
            col_name = X.columns[idx]
            # strings to categorical, then to integer representation
            expr = pl.col(col_name).cast(pl.Categorical).to_physical()
            expr_list.append(expr)
            
    return expr_list
    

def transform(X: pl.LazyFrame, specfile, resultfile): # TODO: scale, save
    """
    Execution phases:
    1. 1-to-1 math expressions: Build an expression list of 1-to-1 math expressions
    (binning, scaling, ...), which is then stacked via .with_columns().
    2. collection: By calling .collect(), polars can "lazily" execute the expression list.
    That means, polars first builds an AST and then finds shortcuts before running.
    3. dummy coding: dc is eager and cannot be handled in the lazy phase, as it alters
    the number of columns ~> We have to handle that separately.
    """
    
    encoders = getTransformSpec(X, specfile)
    timers = np.zeros(3)
    
    # phase 1.1
    expr_list = buildMathExpressions(X, encoders)
    
    for i in range(3):
        t1 = time.time()
        
        # phase 1.2
        transformed = X.with_columns(expr_list)
        
        # phase 2
        transformed = transformed.collect()
        
        # phase 3
        if encoders['dc']:
            dc_col_names = [X.columns[idx] for idx in encoders['dc']]
            transformed = transformed.to_dummies(columns=dc_col_names)
            
        timers[i] = timers[i] + ((time.time() - t1) * 1000) #millisec
        
    print("Elapsed time for transformations in millsec")
    print(timers)
    np.savetxt(resultfile, timers, delimiter="\t", fmt='%f')
    
    return transformed
