"""ACSIncome lake — the US Census successor to Adult.

ACSIncome ("Retiring Adult", Ding et al., NeurIPS 2021) is the 2018 American
Community Survey task built to replace Adult: 1,664,500 people from the 50
states and Puerto Rico, and the same question, whether income exceeds 50K.
Fetched from OpenML (data id 43137); census codes are mapped to readable
values.

    static (stored, queryable)   education, marital_status, race, sex,
                                 age_group, occupation
    predicted (NOT stored)       income ∈ {<=50K, >50K}

Sources are state x class of worker (e.g. "Texas / Federal government"): each
state keeps separate records for public and private employers and the
self-employed, so the lake holds ~400 smaller tables that differ in who they
contain. Both keys define the sources, so neither is a query attribute (a
predicate on them would pin the answer to a few sources); class of worker
stays a stored column the model reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from setup.config import SEED

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OPENML_ID = 43137

STATIC_ATTRS = ["education", "marital_status", "race", "sex",
                "age_group", "occupation"]
PREDICTED_ATTR = "income"
PREDICTED_ATTRS = [PREDICTED_ATTR]

SEX = {1: "Male", 2: "Female"}
MAR = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
       5: "Never married"}
RACE = {1: "White", 2: "Black", 3: "American Indian or Alaska Native",
        4: "American Indian or Alaska Native",
        5: "American Indian or Alaska Native", 6: "Asian",
        7: "Pacific Islander", 8: "Other", 9: "Two or more races"}
COW = {1: "Private for-profit", 2: "Private non-profit", 3: "Local government",
       4: "State government", 5: "Federal government",
       6: "Self-employed (not incorporated)", 7: "Self-employed (incorporated)",
       8: "Family business (unpaid)", 9: "Unemployed"}
STATES = {1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas",
          6: "California", 8: "Colorado", 9: "Connecticut", 10: "Delaware",
          11: "District of Columbia", 12: "Florida", 13: "Georgia", 15: "Hawaii",
          16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa", 20: "Kansas",
          21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland",
          25: "Massachusetts", 26: "Michigan", 27: "Minnesota",
          28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska",
          32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico",
          36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio",
          40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island",
          45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
          48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
          53: "Washington", 54: "West Virginia", 55: "Wisconsin",
          56: "Wyoming", 72: "Puerto Rico"}
# ACS 2018 occupation codes (OCCP) -> major group, by code range.
OCC_GROUPS = [(440, "Management"), (960, "Business and financial"),
              (1240, "Computer and mathematical"),
              (1560, "Architecture and engineering"), (1980, "Science"),
              (2060, "Community and social services"), (2180, "Legal"),
              (2555, "Education"), (2920, "Arts and media"),
              (3550, "Healthcare practitioners"), (3655, "Healthcare support"),
              (3960, "Protective services"), (4160, "Food services"),
              (4255, "Building maintenance"), (4655, "Personal care"),
              (4965, "Sales"), (5940, "Office and administrative"),
              (6130, "Farming and fishing"), (6950, "Construction"),
              (7640, "Installation and repair"), (8990, "Production"),
              (9760, "Transportation"), (9830, "Military")]


def _education(schl: float) -> str:
    s = int(schl)
    if s <= 15:
        return "No high school diploma"
    return {16: "High school", 17: "High school", 18: "Some college",
            19: "Some college", 20: "Associate's", 21: "Bachelor's",
            22: "Master's", 23: "Professional degree",
            24: "Doctorate"}.get(s, "Unknown")


def _occupation(occp: float) -> str:
    for upper, name in OCC_GROUPS:
        if occp <= upper:
            return name
    return "Other"


def load_acs(n_rows: int | None = None) -> pd.DataFrame:
    """ACSIncome 2018 with readable values; `n_rows` samples for a smaller lake."""
    from sklearn.datasets import fetch_openml
    raw = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto",
                       data_home=str(DATA_DIR / "openml")).frame
    if n_rows is not None and len(raw) > n_rows:
        raw = raw.sample(n=n_rows, random_state=SEED)
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({
        "age": raw["AGEP"].astype(int),
        "hours_per_week": raw["WKHP"].astype(int),
        "education_level": raw["SCHL"].astype(int),
        "relationship": raw["RELP"].astype(int),
        "born_abroad": (raw["POBP"] >= 100).map({True: "abroad", False: "US"}),
        "education": raw["SCHL"].map(_education),
        "marital_status": raw["MAR"].astype(int).map(MAR),
        "race": raw["RAC1P"].astype(int).map(RACE),
        "sex": raw["SEX"].astype(int).map(SEX),
        "class_of_worker": raw["COW"].astype(int).map(COW),
        "occupation": raw["OCCP"].map(_occupation),
        "state": raw["ST"].astype(int).map(STATES),
    })
    df["age_group"] = pd.cut(df["age"], bins=[0, 24, 34, 44, 54, 64, 200],
                             labels=["16-24", "25-34", "35-44", "45-54",
                                     "55-64", "65+"]).astype(str)
    df[PREDICTED_ATTR] = np.where(raw["PINCP"] > 50_000, ">50K", "<=50K")
    df["row_id"] = range(len(df))
    return df


def split_train_lake(df: pd.DataFrame, train_frac: float = 0.3):
    """30% of each state × income group trains the models; the rest is the lake."""
    rng = np.random.default_rng(SEED)
    train_idx: list[int] = []
    for _, g in df.groupby(["state", PREDICTED_ATTR], observed=True):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        train_idx.extend(idx[: int(len(idx) * train_frac)].tolist())
    mask = np.zeros(len(df), dtype=bool)
    mask[train_idx] = True
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)


# How the lake can be cut into sources: split name -> the two columns that
# define a source. Cutting on a query attribute means a query asking for several
# of its values has to visit several sources.
SPLITS = {
    "state x class of worker": ("state", "class_of_worker"),
    "state x education": ("state", "education"),
    "state x occupation": ("state", "occupation"),
    "occupation x education": ("occupation", "education"),
}
DEFAULT_SPLIT = "state x class of worker"


def build_lake(n_rows: int | None = None, min_rows: int = 150,
               split: str = DEFAULT_SPLIT):
    """(training_df, {source: df}, truth) with income STRIPPED from every source.

    Sources are cut along `split`; tables under `min_rows` are dropped.
    `truth` maps row_id -> true income and is for EVALUATION ONLY.
    """
    df = load_acs(n_rows=n_rows)
    df_train, df_lake = split_train_lake(df)
    sources = {}
    for (a, b), sub in df_lake.groupby(list(SPLITS[split]), observed=True):
        if len(sub) >= min_rows:
            sources[f"{a} / {b}"] = (sub.drop(columns=[PREDICTED_ATTR, "state"])
                                     .reset_index(drop=True))
    kept = pd.concat([s["row_id"] for s in sources.values()])
    truth = dict(zip(kept.tolist(),
                     df_lake.set_index("row_id").loc[kept, PREDICTED_ATTR].tolist()))
    return df_train, sources, truth


# ── model specification for this lake ────────────────────────────────────────
from setup.models import ModelSpec   # noqa: E402

SPECS = [ModelSpec(
    target=PREDICTED_ATTR,
    num_cols=["age", "hours_per_week", "education_level", "relationship"],
    cat_cols=["education", "marital_status", "race", "sex", "class_of_worker",
              "occupation", "born_abroad"],
    cheap_cols=["education_level", "hours_per_week"], cheap_depth=2,
    max_depth=6,
    kind="gb",
)]
