"""Adult Census Income loader + regional partitioning.

We split the dataset into ~6 source tables keyed by world region (see
config.REGION_MAP). The split is *content-skewed*: one region (North_America)
dwarfs the rest, which is exactly the setting where greedy source ordering
should help PTVD beat random ordering.

A separate held-out subset is reserved for training the proxy + oracle so the
PTVD search itself is run on data the models have never seen — otherwise the
'oracle' would be trivially perfect.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from setup.config import PARTITION_KEY, PREDICTED_ATTR, REGION_MAP, SEED  # noqa: F401

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"   # ptvd/data, not setup/data
CSV_PATH = DATA_DIR / "adult.data"


def _download_if_missing() -> None:
    if CSV_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading Adult to {CSV_PATH}")
    urllib.request.urlretrieve(UCI_URL, CSV_PATH)


def load_adult() -> pd.DataFrame:
    """Load the raw Adult dataframe with whitespace trimmed from string cols."""
    _download_if_missing()
    df = pd.read_csv(
        CSV_PATH, header=None, names=COLUMNS, na_values="?",
        skipinitialspace=True,
    )
    df = df.drop(columns=["fnlwgt"])
    df = df.dropna(subset=["workclass", "occupation", "native_country"]).reset_index(drop=True)
    df["region"] = df[PARTITION_KEY].map(REGION_MAP).fillna("Unknown")
    df["row_id"] = range(len(df))   # globally unique id for oracle lookups
    return df


def split_train_lake(df: pd.DataFrame, train_frac: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve a stratified slice for training proxy/oracle, leave the rest as
    the 'data lake' that PTVD will search over."""
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df))
    # Stratify on (region, income) so every region is represented in both halves.
    train_idx: list[int] = []
    for _, group in df.groupby(["region", "income"]):
        g = group.index.to_numpy().copy()
        rng.shuffle(g)
        cut = int(len(g) * train_frac)
        train_idx.extend(g[:cut].tolist())
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    df_train = df.loc[train_mask].reset_index(drop=True)
    df_lake = df.loc[~train_mask].reset_index(drop=True)
    return df_train, df_lake


def partition_by_region(df_lake: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {region_name: dataframe} — these are the source tables S_1..S_n."""
    return {
        region: sub.drop(columns=["region"]).reset_index(drop=True)
        for region, sub in df_lake.groupby("region")
    }


def shard_sources(
    sources: dict[str, pd.DataFrame],
    target_rows_per_shard: int = 500,
) -> dict[str, pd.DataFrame]:
    """Split each group into roughly equal-size shards (random within group).

    Used to inflate the source count. Shards of the same group are statistically
    homogeneous, so this alone does NOT create ordering signal — it must be
    combined with a heterogeneous partition key.
    """
    rng = np.random.default_rng(SEED)
    out: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        n = len(df)
        k = max(1, n // target_rows_per_shard)
        if k == 1:
            out[name] = df.reset_index(drop=True)
            continue
        idx = np.arange(n)
        rng.shuffle(idx)
        for s, chunk in enumerate(np.array_split(idx, k)):
            out[f"{name}__{s:02d}"] = df.iloc[chunk].reset_index(drop=True)
    return out


def partition_by(df_lake: pd.DataFrame, key: str) -> dict[str, pd.DataFrame]:
    """Partition the lake by an arbitrary categorical column.

    Partitioning by `occupation` (or another attribute with strong income
    variation) yields HETEROGENEOUS sources — different sources have very
    different P(income>50K) — which is the setting where PTVD's stats-driven
    ordering pays off. This mirrors a real data lake where each table comes
    from a different institution with its own population skew.
    """
    drop = [c for c in ("region",) if c in df_lake.columns]
    return {
        str(val): sub.drop(columns=drop).reset_index(drop=True)
        for val, sub in df_lake.groupby(key)
    }


def build_lake(
    partition_key: str = "occupation",
    target_rows_per_shard: int | None = 500,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[int, str]]:
    """Returns (training_df, {source_name: source_df}, truth).

    CRITICAL: the predicted attribute (`income`) is STRIPPED from every source
    table. In the PTVD data model predicted attributes are *not stored* — they
    are produced by a model at query time. Leaving the true label in the source
    would let retrieval and the evidence measures read the answer directly.

    `truth` maps row_id -> true income and is for EVALUATION ONLY; it must
    never be consulted by the pipeline.

    partition_key="occupation" (default) → heterogeneous, income-skewed sources.
    target_rows_per_shard=None           → one source per partition value.
    """
    df = load_adult()
    df_train, df_lake = split_train_lake(df)
    truth = dict(zip(df_lake["row_id"], df_lake[PREDICTED_ATTR]))

    sources = partition_by(df_lake, partition_key)
    if target_rows_per_shard is not None:
        sources = shard_sources(sources, target_rows_per_shard)
    sources = {
        name: src.drop(columns=[PREDICTED_ATTR])
        for name, src in sources.items()
    }
    return df_train, sources, truth


if __name__ == "__main__":
    df_train, sources, truth = build_lake()
    print(f"training set: {len(df_train)} rows")
    print(f"sources: {len(sources)}   (predicted attr '{PREDICTED_ATTR}' stripped)")
    for name, src in sorted(sources.items(), key=lambda kv: -len(kv[1]))[:5]:
        pos = np.mean([truth[r] == ">50K" for r in src["row_id"]])
        print(f"  {name:<22} {len(src):>5} rows   true P(>50K)={pos:.3f}")
    assert all(PREDICTED_ATTR not in s.columns for s in sources.values())
    print("OK: no source exposes the predicted attribute")


# ── uniform lake interface ───────────────────────────────────────────────────
# Static attributes queries may use. `occupation` is excluded: it is the
# partition key, so a predicate on it pins all matching rows to one source and
# source selection becomes definitionally irrelevant.
STATIC_ATTRS = ["education", "marital_status", "race", "sex",
                "workclass", "native_country"]
PREDICTED_ATTRS = [PREDICTED_ATTR]

# ── model specification for this lake ────────────────────────────────────────
from setup.models import ModelSpec   # noqa: E402

SPECS = [ModelSpec(
    target=PREDICTED_ATTR,
    num_cols=["age", "education_num", "capital_gain", "capital_loss",
              "hours_per_week"],
    cat_cols=["workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex"],
    cheap_cols=["capital_gain"], cheap_depth=1,
    max_depth=6,
    kind="gb",
)]
