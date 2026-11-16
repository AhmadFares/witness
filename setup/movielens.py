"""MovieLens 1M lake — a harder setting than Adult.

Why a second dataset: Adult has ONE binary predicted attribute, so the label
space has two values, CEv is pinned near a constant, and the conditional
estimator is barely exercised. MovieLens gives TWO predicted attributes, both
MULTI-class, over many more sources.

    static (stored, queryable)   gender, age_group, occupation, region, decade
    predicted (NOT stored)       rating_band  ∈ {low, mid, high}
                                 genre        ∈ 8 primary genres

Both predicted columns are stripped from the sources exactly as `income` is in
the Adult lake — in the PTVD data model predicted attributes are produced by a
model at query time, never read.

Sources are region × decade, following the TVD paper's MovieLens split.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from setup.config import SEED

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ML_DIR = DATA_DIR / "ml-1m"

STATIC_ATTRS = ["gender", "age_group", "occupation", "region", "decade"]
PREDICTED_ATTRS = ["rating_band", "genre"]

# MovieLens ships numeric codes; these are the documented meanings.
AGE = {1: "under18", 18: "18-24", 25: "25-34", 35: "35-44",
       45: "45-49", 50: "50-55", 56: "56+"}
OCC = {0: "other", 1: "academic", 2: "artist", 3: "clerical", 4: "student",
       5: "customer-service", 6: "health", 7: "executive", 8: "farmer",
       9: "homemaker", 10: "student-k12", 11: "lawyer", 12: "programmer",
       13: "retired", 14: "sales", 15: "scientist", 16: "self-employed",
       17: "engineer", 18: "craftsman", 19: "unemployed", 20: "writer"}
# US zip first digit -> broad region
REGION = {"0": "Northeast", "1": "Northeast", "2": "MidAtlantic",
          "3": "Southeast", "4": "Midwest", "5": "Midwest",
          "6": "Central", "7": "South", "8": "Mountain", "9": "West"}
GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
          "Horror", "Romance", "Sci-Fi", "Thriller"]


def _download_if_missing() -> None:
    if ML_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zpath = DATA_DIR / "ml-1m.zip"
    if not zpath.exists():
        print(f"[data] downloading MovieLens 1M to {zpath}")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)


def load_movielens(n_rows: int | None = 200_000) -> pd.DataFrame:
    """Join ratings x users x movies into one flat table."""
    _download_if_missing()
    kw = dict(sep="::", engine="python", encoding="latin-1", header=None)

    users = pd.read_csv(ML_DIR / "users.dat", names=
                        ["user_id", "gender", "age", "occ", "zip"], **kw)
    movies = pd.read_csv(ML_DIR / "movies.dat", names=
                         ["movie_id", "title", "genres"], **kw)
    ratings = pd.read_csv(ML_DIR / "ratings.dat", names=
                          ["user_id", "movie_id", "rating", "ts"], **kw)

    users["age_group"] = users["age"].map(AGE).fillna("other")
    users["occupation"] = users["occ"].map(OCC).fillna("other")
    users["region"] = users["zip"].astype(str).str[0].map(REGION).fillna("Other")

    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["year"] = pd.to_numeric(year, errors="coerce")
    movies = movies.dropna(subset=["year"])
    movies["decade"] = (movies["year"] // 10 * 10).astype(int).astype(str) + "s"

    # primary genre = first listed genre that is one of the common ones
    def primary(g):
        for part in str(g).split("|"):
            if part in GENRES:
                return part
        return None
    movies["genre"] = movies["genres"].map(primary)
    movies = movies.dropna(subset=["genre"])

    df = (ratings
          .merge(users[["user_id", "gender", "age_group", "occupation", "region"]],
                 on="user_id")
          .merge(movies[["movie_id", "decade", "genre", "year"]], on="movie_id"))

    # rating 1-5 -> three bands, so the predicted attribute is multi-class
    df["rating_band"] = pd.cut(df["rating"], bins=[0, 2, 3, 5],
                               labels=["low", "mid", "high"]).astype(str)

    if n_rows is not None and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=SEED)
    df = df.reset_index(drop=True)
    df["row_id"] = range(len(df))
    return df


def split_train_lake(df: pd.DataFrame, train_frac: float = 0.3):
    """Stratified holdout so the models never see lake rows."""
    rng = np.random.default_rng(SEED)
    train_idx: list[int] = []
    for _, g in df.groupby(["region", "rating_band"], observed=True):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        train_idx.extend(idx[: int(len(idx) * train_frac)].tolist())
    mask = np.zeros(len(df), dtype=bool)
    mask[train_idx] = True
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)


def _attach_history(df_train: pd.DataFrame, df_lake: pd.DataFrame):
    """Add movie/user rating tendencies as STATIC stored columns.

    Demographics alone do not predict how someone rates a film — a model built
    on them is no better than guessing the majority class. The real signal is
    that some movies are better liked and some users rate more generously.
    Both are ordinary metadata a movies/users table would carry, so they are
    static attributes, not leakage: they are computed from the TRAINING split
    only and never involve the lake row's own label.
    """
    gm = df_train["rating"].mean()
    movie_avg = df_train.groupby("movie_id")["rating"].mean()
    user_avg = df_train.groupby("user_id")["rating"].mean()
    movie_pop = df_train.groupby("movie_id").size()

    for d in (df_train, df_lake):
        d["movie_avg"] = d["movie_id"].map(movie_avg).fillna(gm)
        d["user_avg"] = d["user_id"].map(user_avg).fillna(gm)
        d["movie_pop"] = d["movie_id"].map(movie_pop).fillna(0)
    return df_train, df_lake


def build_lake(min_rows: int = 150, n_rows: int | None = 200_000,
               partition: tuple[str, str] = ("region", "decade")):
    """(training_df, {source: df}, truth) with predicted attributes STRIPPED.

    `truth` maps row_id -> {attr: true value} for EVALUATION ONLY.
    partition=("region","decade")      → ~58 sources (default, TVD-style)
    partition=("occupation","decade")  → ~150 sources (the scale variant)
    Tiny sources are dropped so statistics mean something.
    """
    df = load_movielens(n_rows=n_rows)
    df_train, df_lake = split_train_lake(df)
    df_train, df_lake = _attach_history(df_train, df_lake)

    truth = {
        int(r): {a: rec[a] for a in PREDICTED_ATTRS}
        for r, rec in zip(df_lake["row_id"], df_lake.to_dict("records"))
    }

    sources = {}
    for key, sub in df_lake.groupby(list(partition), observed=True):
        if len(sub) < min_rows:
            continue
        name = "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        sources[name] = (
            sub.drop(columns=PREDICTED_ATTRS + ["rating"])
               .reset_index(drop=True))
    return df_train, sources, truth


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    df_train, sources, truth = build_lake()
    print(f"training set : {len(df_train)} rows")
    print(f"sources      : {len(sources)}  (predicted attrs stripped)")
    tot = sum(len(s) for s in sources.values())
    print(f"lake rows    : {tot}")
    for a in PREDICTED_ATTRS:
        vals = pd.Series([t[a] for t in truth.values()])
        print(f"  {a:<12} {vals.nunique()} classes: {dict(vals.value_counts().head(5))}")
    print("\nlargest sources:")
    for n, s in sorted(sources.items(), key=lambda kv: -len(kv[1]))[:6]:
        hi = np.mean([truth[r]["rating_band"] == "high" for r in s["row_id"]])
        print(f"  {n:<22} {len(s):>6} rows   P(high)={hi:.3f}")
    assert all(not set(PREDICTED_ATTRS) & set(s.columns) for s in sources.values())
    print("\nOK: no source exposes a predicted attribute")


# ── model specifications for this lake ───────────────────────────────────────
from setup.models import ModelSpec   # noqa: E402

# Features are STATIC attributes only: a model cannot read another predicted
# attribute, since predicted attributes are not stored.
_CAT = ["gender", "age_group", "occupation", "region", "decade"]
_NUM = ["year", "movie_avg", "user_avg", "movie_pop"]

SPECS = [
    ModelSpec(target="rating_band", num_cols=_NUM, cat_cols=_CAT,
              cheap_cols=["movie_avg", "user_avg"], cheap_depth=3, max_depth=8,
              kind="gb"),
    ModelSpec(target="genre", num_cols=_NUM, cat_cols=_CAT,
              cheap_cols=["year", "movie_avg"], cheap_depth=3, max_depth=8,
              kind="gb"),
]
