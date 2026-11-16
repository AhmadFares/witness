"""MovieLens 25M joined with IMDb — ratings, with movie themes to predict.

Each row is one rating from MovieLens 25M. The rating and the movie's
catalogue data are stored: MovieLens genres, and IMDb's release year, runtime,
average rating and number of votes (joined through MovieLens' imdbId links).
What no catalogue stores is what a film is LIKE: whether it is dark,
suspenseful or funny. An LLM would have to read the plot or the reviews; here
an expensive model predicts it from the stored metadata. The true values come
from the MovieLens tag genome (a theme is "yes" for the 30% of movies most
relevant to it) and are used for evaluation only.

    static (stored, queryable)   genre, user_rating, runtime, imdb_rating,
                                 popularity
    predicted (NOT stored)       dark, suspenseful, funny ∈ {yes, no}

Themes belong to the movie, so each is predicted once per movie and reused
for the movie's other ratings (ModelSpec.entity = "movie_id").

Models are trained on 30% of the MOVIES and the lake holds the ratings of the
other 70%, so the model always judges films it has never seen; splitting
ratings instead would let it recognise a film it was trained on.

Sources are year of rating x decade of release. Both define the sources, so
neither is a query attribute.

Data: MovieLens 25M (GroupLens); IMDb non-commercial datasets (title.basics,
title.ratings), for non-commercial research use.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from setup.config import SEED

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ML_DIR = DATA_DIR / "ml-25m"
IMDB_DIR = DATA_DIR / "imdb"
N_ROWS = 1_500_000                     # ratings kept (sampled from 25M)
CACHE = DATA_DIR / f"ml25m_imdb_{N_ROWS}.parquet"

THEMES = {"dark": "dark", "suspenseful": "suspense", "funny": "funny"}
PREDICTED_ATTRS = list(THEMES)
STATIC_ATTRS = ["genre", "user_rating", "runtime", "imdb_rating", "popularity"]
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
          "Western"]
FLAGS = [f"g_{g}" for g in GENRES]


def _imdb(tconsts: set[str]) -> pd.DataFrame:
    kw = dict(sep="\t", na_values="\\N", quoting=csv.QUOTE_NONE, dtype=str)
    basics = pd.concat(
        c[c.tconst.isin(tconsts)] for c in pd.read_csv(
            IMDB_DIR / "title.basics.tsv.gz", usecols=["tconst", "startYear",
                                                       "runtimeMinutes"],
            chunksize=1_000_000, **kw))
    ratings = pd.read_csv(IMDB_DIR / "title.ratings.tsv.gz", **kw)
    out = basics.merge(ratings, on="tconst", how="left").set_index("tconst")
    return out.apply(pd.to_numeric, errors="coerce")


def _movies() -> pd.DataFrame:
    """One row per movie of the tag genome: stored metadata, theme labels."""
    tags = pd.read_csv(ML_DIR / "genome-tags.csv").set_index("tagId")["tag"]
    want = {tid: name for name, tag in THEMES.items()
            for tid in tags.index[tags == tag]}
    rel = pd.concat(c[c.tagId.isin(want)] for c in pd.read_csv(
        ML_DIR / "genome-scores.csv", chunksize=2_000_000))
    rel = rel.pivot(index="movieId", columns="tagId", values="relevance").rename(columns=want)

    movies = pd.read_csv(ML_DIR / "movies.csv").set_index("movieId").loc[rel.index]
    links = pd.read_csv(ML_DIR / "links.csv").set_index("movieId").loc[rel.index]
    tconst = "tt" + links["imdbId"].astype(int).astype(str).str.zfill(7)
    imdb = _imdb(set(tconst)).reindex(tconst.values)

    m = pd.DataFrame(index=rel.index)
    m.index.name = "movie_id"
    title_year = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0],
                               errors="coerce")
    m["year"] = imdb["startYear"].to_numpy()
    m["year"] = m["year"].fillna(title_year)
    m["runtime_min"] = imdb["runtimeMinutes"].to_numpy()
    m["imdb_avg"] = imdb["averageRating"].to_numpy()
    m["imdb_votes"] = imdb["numVotes"].to_numpy()
    for g, f in zip(GENRES, FLAGS):
        m[f] = movies["genres"].str.contains(g, regex=False).astype(int)
    m = m.dropna(subset=["year", "runtime_min", "imdb_avg", "imdb_votes"])

    m["decade"] = (m["year"] // 10 * 10).astype(int).astype(str) + "s"
    m["genre"] = movies.loc[m.index, "genres"].str.split("|").str[0]
    m["runtime"] = pd.cut(m["runtime_min"], [0, 89, 120, 10_000],
                          labels=["short", "standard", "long"]).astype(str)
    m["imdb_rating"] = pd.cut(m["imdb_avg"], [0, 5.99, 6.99, 7.99, 10],
                              labels=["below 6", "6-7", "7-8", "8+"]).astype(str)
    m["popularity"] = pd.cut(m["imdb_votes"], [0, 9_999, 99_999, 10**9],
                             labels=["niche", "known", "popular"]).astype(str)
    for name in THEMES:
        r = rel.loc[m.index, name]
        m[name] = np.where(r >= r.quantile(0.7), "yes", "no")
    rng = np.random.default_rng(SEED)
    m["train"] = rng.random(len(m)) < 0.3
    return m


def load(n_rows: int = N_ROWS) -> pd.DataFrame:
    """Sampled ratings joined with their movie. Built once, then read from a
    Parquet cache (the join over 25M ratings and IMDb takes about a minute)."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    m = _movies()
    r = pd.read_csv(ML_DIR / "ratings.csv", dtype={"userId": "int32", "movieId": "int32",
                                                   "rating": "float32", "timestamp": "int64"})
    r = r[r.movieId.isin(m.index)]
    r = r.sample(n=min(n_rows, len(r)), random_state=SEED).reset_index(drop=True)
    r = r.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    r["rating_year"] = pd.to_datetime(r.pop("timestamp"), unit="s").dt.year
    r["user_rating"] = pd.cut(r["rating"], [0, 2.5, 3.5, 5],
                              labels=["disliked", "neutral", "liked"]).astype(str)
    df = r.join(m, on="movie_id")
    df["row_id"] = np.arange(len(df))
    df.to_parquet(CACHE, index=False)
    return df


class _RowTruth(Mapping):
    """row_id -> {theme: true value}, stored per movie instead of per row."""

    def __init__(self, rows: pd.DataFrame, per_movie: dict):
        self._movie = dict(zip(rows["row_id"].tolist(), rows["movie_id"].tolist()))
        self._per_movie = per_movie

    def __getitem__(self, row_id):
        return self._per_movie[self._movie[row_id]]

    def __iter__(self):
        return iter(self._movie)

    def __len__(self):
        return len(self._movie)


# How the lake can be cut into sources: split name -> the two columns that
# define a source. Cutting on a query attribute means a query asking for several
# of its values has to visit several sources.
SPLITS = {
    "year of rating x decade of release": ("rating_year", "decade"),
    "genre x decade of release": ("genre", "decade"),
    "genre x year of rating": ("genre", "rating_year"),
    "decade of release x IMDb rating": ("decade", "imdb_rating"),
}
DEFAULT_SPLIT = "year of rating x decade of release"


def build_lake(min_rows: int = 150, split: str = DEFAULT_SPLIT):
    """(training_df, {source: df}, truth) with the themes STRIPPED.

    The training frame has one row per training movie (themes are a property
    of the movie). `truth` is for EVALUATION ONLY.
    """
    df = load()
    lake = df[~df["train"]]
    train = df[df["train"]].drop_duplicates("movie_id").reset_index(drop=True)
    per_movie = (lake.drop_duplicates("movie_id").set_index("movie_id")[PREDICTED_ATTRS]
                 .to_dict("index"))
    truth = _RowTruth(lake, per_movie)
    drop = PREDICTED_ATTRS + ["train", "rating_year", "decade"]
    sources = {}
    for (a, b), sub in lake.groupby(list(SPLITS[split]), observed=True):
        if len(sub) >= min_rows:
            name = f"{a}_{b}" if split == DEFAULT_SPLIT else f"{a} / {b}"
            sources[name] = sub.drop(columns=drop).reset_index(drop=True)
    return train, sources, truth


# ── model specifications for this lake ───────────────────────────────────────
from setup.models import ModelSpec   # noqa: E402

_NUM = ["year", "runtime_min", "imdb_avg", "imdb_votes"] + FLAGS
_CHEAP = {"dark": ["g_Horror", "g_Thriller"], "suspenseful": ["g_Thriller", "g_Mystery"],
          "funny": ["g_Comedy", "g_Drama"]}

SPECS = [ModelSpec(target=t, num_cols=_NUM, cat_cols=["genre"],
                   cheap_cols=_CHEAP[t], cheap_depth=2, max_depth=6,
                   entity="movie_id", kind="gb")
         for t in PREDICTED_ATTRS]
