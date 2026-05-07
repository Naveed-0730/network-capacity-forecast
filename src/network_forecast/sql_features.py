"""
SQL feature engineering for network telemetry.

We use DuckDB so the SQL is portable and runs against Parquet files locally,
but every query here is written in standard ANSI-ish SQL that translates
cleanly to Snowflake, BigQuery, or Redshift — the kind of warehouse a Sky
data team would actually use.

The point of this module is to demonstrate that feature engineering at scale
belongs in the warehouse, not in pandas. Anything joining 10M+ rows of
telemetry to subscriber data should never touch a Python kernel until the
aggregations are done.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


# Daily aggregates — the kind of feature you'd build for a capacity-planning
# model that operates at daily granularity.
DAILY_AGGREGATES_SQL = """
SELECT
    pop,
    DATE_TRUNC('day', timestamp) AS day,
    AVG(traffic_gbps)              AS mean_traffic_gbps,
    MAX(traffic_gbps)              AS peak_traffic_gbps,
    QUANTILE_CONT(traffic_gbps, 0.95) AS p95_traffic_gbps,
    STDDEV_SAMP(traffic_gbps)      AS std_traffic_gbps,
    COUNT(*)                       AS n_observations
FROM read_parquet('{data_path}')
GROUP BY pop, day
ORDER BY pop, day
"""

# Weekly busy-hour analysis — the "what's my real worst case" view for
# capacity planners. Network ops cares about the 95th percentile of the
# busiest hour each week, not the average.
BUSY_HOUR_SQL = """
WITH hourly AS (
    SELECT
        pop,
        DATE_TRUNC('week', timestamp) AS week,
        EXTRACT(HOUR FROM timestamp)  AS hour_of_day,
        AVG(traffic_gbps)             AS avg_gbps
    FROM read_parquet('{data_path}')
    GROUP BY pop, week, hour_of_day
)
SELECT
    pop,
    week,
    hour_of_day AS busy_hour,
    avg_gbps    AS busy_hour_traffic_gbps
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY pop, week ORDER BY avg_gbps DESC) AS rk
    FROM hourly
)
WHERE rk = 1
ORDER BY pop, week
"""

# Lag-based features for ML training. Computing these in SQL avoids having
# to roundtrip the full table to Python — only the final feature matrix.
LAG_FEATURES_SQL = """
SELECT
    pop,
    timestamp,
    traffic_gbps                                         AS y,
    LAG(traffic_gbps,   1) OVER w AS lag_1h,
    LAG(traffic_gbps,  24) OVER w AS lag_1d,
    LAG(traffic_gbps, 168) OVER w AS lag_1w,
    AVG(traffic_gbps) OVER (
        PARTITION BY pop
        ORDER BY timestamp
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    )                                                    AS rmean_24h,
    AVG(traffic_gbps) OVER (
        PARTITION BY pop
        ORDER BY timestamp
        ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
    )                                                    AS rmean_1w,
    EXTRACT(HOUR     FROM timestamp) AS hour,
    EXTRACT(DOW      FROM timestamp) AS day_of_week,
    EXTRACT(MONTH    FROM timestamp) AS month
FROM read_parquet('{data_path}')
WINDOW w AS (PARTITION BY pop ORDER BY timestamp)
ORDER BY pop, timestamp
"""


def run_query(sql_template: str, data_path: str | Path) -> pd.DataFrame:
    """Execute a templated query against the parquet dataset and return a DataFrame."""
    data_path = str(Path(data_path).resolve())
    sql = sql_template.format(data_path=data_path)
    logger.info("Running DuckDB query against %s", data_path)
    return duckdb.query(sql).to_df()


def build_daily_aggregates(data_path: str | Path) -> pd.DataFrame:
    """Daily mean/peak/p95 traffic per PoP — a stakeholder-ready summary."""
    return run_query(DAILY_AGGREGATES_SQL, data_path)


def build_busy_hour_summary(data_path: str | Path) -> pd.DataFrame:
    """Weekly busy-hour table — the canonical capacity-planning view."""
    return run_query(BUSY_HOUR_SQL, data_path)


def build_lag_features(data_path: str | Path) -> pd.DataFrame:
    """Full lag-feature matrix for downstream ML training."""
    return run_query(LAG_FEATURES_SQL, data_path)
