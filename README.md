# Current Short-Term Notes
I am unable to initialize the dataset for T11, as it is so big (>6GB) that my VM crashes when running wiki_prep.py.
I am also unable to initialize the dataset for T15, as I need to download it manually, but the WeTransfer link is outdated.

# General

## Purpose
FTBench currently implements only a subset of the 15 transformations (T1 through T15) on only a certain number of libraries. Our goal is to extend FTBench by completing transformations for the currently implemented libraries and/or adding transformations in new libraries.

E.g. for Keras, we could add T4-T9 and T13-T15. Or we could explore Pandas, Polars, etc.

## Setup
When following the setup in UPLIFT/README.md, after executing UPLIFT/system_setup.md, create and activate .venv using
```
python3.9 -m venv .venv
source .venv/bin/activate
```
Then do the rest everything as mentioned in UPLIFT/README.md.

## Links
- [issue](https://issues.apache.org/jira/browse/SYSTEMDS-3645)
- [rep. repo](https://github.com/damslab/reproducibility/tree/master/vldb2022-UPLIFT-p2528/FTBench)
- [paper](https://www.vldb.org/pvldb/vol15/p2929-phani.pdf)

# Plan / Doing
- [x] Initial call
- [x] Create and set up repo with current transformations and libraries
- [x] Analyze current implementations
- [x] Plan what implementations to add
- [ ] Learn about Polars
- [ ] Set up Polars
- [ ] T1: Polars


## Current implementations
|Transformation|Dask                         |Keras |NimbusML                                     |SKLearn|SparkML|SystemDS|
|--------------|-----------------------------|------|---------------------------------------------|-------|-------|--------|
|T1            |✅                            |✅     |-                                            |✅      |✅      |✅       |
|T2            |✅                            |✅     |-                                            |✅      |✅      |✅       |
|T3            |✅                            |✅     |✅                                            |✅      |✅      |✅       |
|T4            |-                            |-     |-                                            |✅      |-      |✅       |
|T5            |-                            |-     |-                                            |✅      |-      |✅       |
|T6            |-                            |-     |-                                            |✅      |-      |✅       |
|T7            |-                            |-     |-                                            |✅      |-      |✅       |
|T8            |-                            |-     |-                                            |✅      |-      |✅       |
|T9            |-                            |-     |-                                            |✅      |✅      |✅       |
|T10           |-                            |✅     |-                                            |✅      |-      |✅       |
|T11           |-                            |✅     |-                                            |-      |-      |✅       |
|T12           |-                            |✅     |-                                            |✅      |-      |✅       |
|T13           |-                            |-     |-                                            |✅      |-      |✅       |
|T14           |-                            |-     |-                                            |✅      |-      |✅       |
|T15           |-                            |-     |-                                            |✅      |✅      |✅       |

## Implementation Plan
The benchmark should implement the current sota to be of value. We should therefore add Polars to the benchmark, which is written in Rust, heavily multi-threaded (in contrast to Pandas) and therefore very fast ~> It's an aggressive competitor for SystemDS and beating it will be difficult.