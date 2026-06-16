# Polars vs. scikit-learn `transformUtils`: open problems

Short comparison of my [polars/transformUtils.py](transformUtils.py) against the reference [scikit-learn/transformUtils.py](../scikit-learn/transformUtils.py), with a focus on (a) whether the two implementations perform the *same transformations* and (b) whether their timed sections measure the *same work*, so that runtimes are comparable. Findings are mostly based on `T1_*` (adult, `adult_spec2.json`), plus a review of all specs in `systemds/specs/`.

## 1. KBinsDiscretizer is applied to *all* numeric columns, not just `encoders['bins']`

In sklearn, `num_pipe` is **sequential**:

```python
numeric = list(X.select_dtypes(include=np.float64).columns)  # == pt ∪ bins
num_pipe = Pipeline([('selector', ColumnSelector(numeric))])
if scale: num_pipe.steps.append(['normalize', StandardScaler()])
if isBin: num_pipe.steps.append(['biner', KBinsDiscretizer(...)])
```

Since `numeric` = `pt` ∪ `bins` and the pipeline is sequential, `KBinsDiscretizer` is applied to **every** numeric column that reaches it -- including `pt` columns, not just the ones listed in `encoders['bins']`.

Example: `adult_spec2.json` has `pt = [4]`, `bins = [0, 2, 10, 11, 12]`and `isBinDC` is `True` (all bin columns are also in `dummycode`), so `postBin = 'onehot'` &rarr; sklearn's `num_pipe` one-hot-encodes **6** columns into 5 bins each (30 output columns total), where 5 of those 30 columns come from binning+one-hot-encoding the passthrough column. My Polars implementation treats them separately and therefore producted **5 fewer columns** than sklearn's, skipping the (in sklearn's case, ultimately discarded/overwritten) extra binning work.

This is likely a  bug/quirk of the sklearn reference (acknowledged partially by the "TODO: support column specific methods and numbins" / "TODO: support mixed encoders" comments), but it means that (a) output shapes differ between the two implementations, and (b) sklearn performs strictly more transformation work in this case.

> **Decision** Keep the current polars implementation, i.e. bin only `encoders['bins']`, leave `pt` columns as passthrough.

## 2. `scale=True` scales categorical/binned output in sklearn, but polars only scales passthrough columns (resolved)

> **Implemented** See "Implemented solution" below.

sklearn implementation when `scale=True`:
- `num_pipe` gets a `StandardScaler()` (mean+std) inserted **before** the binning step, applied to all of `pt ∪ bins`.
- `cat_pipe` gets `StandardScaler(with_mean=False)` appended **after** the one-hot/ordinal encoder, scaling the encoded categorical output (which may be is hundreds of one-hot columns).

Both specs that exercise `scale=True` (`kdd_spec1.json`, `criteo_spec2.json`) have `isBin=True`. As mentioned in the first issue, `KBinsDiscretizer` is applied to **both** `pt` and `bins`, overwriting `StandardScaler` with bin indices/one-hot &rarr; `num_pipe`'s `StandardScaler` has no effect on sklearn's output for these two specs and is just dead computation.

So, the effect of `scale=True` comes just from `cat_pipe`'s `StandardScaler(with_mean=False)` and standardizes the recode/dummy-coded categorical output &rarr; So "scale" in this benchmark suite semantically means "standardize the encoded categorical/dummy output", which is exactly what `cat_pipe` does.

### Implemented solution

I added a "phase 4" to `transform()`, run after `to_dummies` (phase 3), only
when `scale=True`:

- `rc_scale_cols`: `rc` columns that are neither `bins` nor also `dc` (since `recode ⊆ dummycode` in all current specs, this set is empty in practice, but included for correctness in case of future rc-only specs).
- `dc_scale_cols`: `to_dummies`-generated columns whose source column is in `dc` but not in `bins`. Bin-derived `dc` columns are excluded, matching sklearn.
- `scale_cols = rc_scale_cols + dc_scale_cols`: scaled, matching `StandardScaler(with_mean=False)`.

---

## Deferred (not being addressed for now)

## 3. Mixed recode/dummycode handling

sklearn's `cat_pipe` applies **one** encoder to *all* `catCols` (`rc` and `dc`) and is constructed like so:

```python
if isDC:
    cat_pipe.steps.append(['onehot', OneHotEncoder()])
elif isRC:
    cat_pipe.steps.append(['recode', OrdinalEncoder()])
```

If any `dc` column exists, every categorical column (including pure-`rc` ones) gets one-hot encoded -- as is also stated via "TODO: support mixed encoders". My Polars implementation instead correctly applies ordinal encoding to `rc` columns and `to_dummies` to `dc` columns independently.

However, I don't think this is a problem at the moment. I let AI check all specs in `systemds/specs/`: in every spec where `recode` and `dummycode` are non-empty, `recode ⊆ dummycode` &rarr; so there is currently no "rc-only" column that would be mis-encoded by sklearn's shortcut.

## 4. Feature hashing (`hash`/`fh`) — unsupported in both
