# Architecture & design decisions

This doc captures the *why* behind the choices in this repo. The README
covers what the project does and how to run it; this is the back-of-the-
napkin discussion you'd have with another engineer in a code review.

## Why an `ABC` for forecasters

Three motivations:

1. **Champion-challenger.** The whole point of `walk_forward_eval` is to
   compare models on identical data and identical metrics. That only works
   if every model exposes the same surface — `fit(series)` and
   `predict(horizon)`. An `ABC` enforces that contract at class-definition
   time rather than at runtime when a missing method explodes mid-training.

2. **Dashboard genericity.** The Streamlit app passes a model name to a
   factory and gets back something it can call `fit/predict` on. It does
   not know — and should not know — whether the underlying model is
   statsmodels or LightGBM or a Prophet wrapper. New models slot in with
   zero dashboard changes.

3. **Test surface area.** Unit tests can target the protocol (`predict
   before fit raises NotFittedError`) once, parameterised across all
   concrete classes, instead of replicating the same test three times.

## Why walk-forward, not k-fold

K-fold cross-validation on time series data leaks future into past — fold
3 might train on observations from 2024 and validate on observations from
2023. The model has effectively seen the answer. Walk-forward
(rolling-origin) keeps the temporal ordering intact: train on `[0, t]`,
test on `(t, t+horizon]`, then advance `t`.

The cost is fewer effective folds (you can't reuse early observations as
validation), but the metrics you get back actually mean something. There's
no point reporting an MAE of 5.2 if the validation procedure is
fundamentally optimistic.

## Why MAD, not std, for anomaly detection

A 3σ threshold built on `mean ± std` of historical residuals breaks down
the moment you have real anomalies in the calibration set: each big
deviation pumps up the std, which raises the threshold, which makes future
real anomalies easier to miss. It's a doom loop.

Median + MAD (median absolute deviation, scaled by 1.4826 to be a
consistent estimator of σ under normality) is robust. A single outlier
moves the median by approximately one observation's worth of weight rather
than dragging it proportionally. The MAD itself is the median of absolute
deviations, so anomalies get downweighted automatically.

This is the classic non-parametric move: when you can't trust the tails of
your distribution, score with quantile-based statistics instead of moments.

## Why DuckDB for SQL features

Three reasons over loading Parquet into pandas and aggregating there:

1. **Demonstrates SQL competence in a SQL-shaped way.** The JD lists SQL
   as a separate skill from Python. Showing a `GROUP BY ... QUANTILE_CONT`
   query in actual SQL is more credible than a `pandas.groupby().quantile`
   call dressed up as "SQL-equivalent".

2. **The queries are portable.** Every query in `sql_features.py` is
   ANSI-ish SQL that runs against Snowflake, BigQuery, or Redshift with
   minimal changes. DuckDB is the local execution engine; the SQL itself
   is the artifact a Sky data engineer could lift straight into production.

3. **It scales.** 70k rows is small enough for pandas, but the same code
   handles 70M without rewriting — DuckDB pushes filters and aggregations
   down to the Parquet reader and never materialises the full table in
   memory.

## Why LightGBM lost to seasonal naive

The honest answer is that the synthetic data has very stable weekly
seasonality and not much else. The features `lag_168`, `lag_24`, `lag_1`,
`hour`, `dow` essentially recover what the seasonal naive baseline already
encodes, but with extra parameters fit to noise. Add real-world drivers
(subscriber growth shocks, sports fixtures, content release windows,
weather), and LightGBM would start to dominate because those signals
*can't* be encoded by "look at the same hour last week".

This is a useful result to report rather than hide. Reaching for the
fanciest model and assuming it'll win is exactly the failure mode that
seasoned data scientists in production environments learn to avoid.

## What I'd do next in production

- **Trend-adjusted baseline.** The seasonal-naive model doesn't account
  for the YoY subscriber growth in the data. A simple fix:
  `y_t = y_{t-168} * (1 + weekly_growth_rate)`. This would reduce false
  positives in the anomaly detector.

- **Hierarchical reconciliation.** The four PoPs sum to a national
  total. Forecasting them independently can produce inconsistent
  aggregates. Tools like `hts` (or a custom MinT reconciliation step)
  ensure the bottom-up forecast matches a top-down one.

- **Probabilistic forecasts everywhere.** Right now only SARIMA returns
  intervals. LightGBM-quantile or conformal prediction would give the
  capacity-planning team the percentile bands they actually need to
  decide upgrade timing.

- **Drift monitoring.** A scheduled job that compares the rolling-window
  distribution of residuals to the calibration distribution, with a
  Slack alert when KS-distance crosses a threshold. Cheap to build,
  catches model degradation before customers notice.
