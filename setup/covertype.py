"""Covertype lake — the scale dataset (581k rows, 7-class predicted attribute).

Third dataset, chosen for scale and for a predicted attribute with 7 classes:
forest cover type, which in this data model is NOT stored — a model produces
it at query time from terrain features. Static attributes are coarse terrain
descriptors a catalogue would store (bands and directions), and sources are
wilderness_area x soil_type — genuinely heterogeneous cells of the forest,
yielding 100+ sources.

Public, no credentials: sklearn.datasets.fetch_covtype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from setup.config import SEED
from setup.models import ModelSpec

WILDERNESS = ["Rawah", "Neota", "ComanchePeak", "CachePoudre"]
COVER = {1: "Spruce-Fir", 2: "Lodgepole", 3: "Ponderosa", 4: "Cottonwood",
         5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}

STATIC_ATTRS = ["elevation_band", "aspect_dir", "slope_band", "hydro_band"]
PREDICTED_ATTRS = ["cover_type"]

NUM_COLS = ["elevation", "aspect", "slope", "hydro_dist", "road_dist",
            "hillshade_noon", "fire_dist"]
CAT_COLS = ["wilderness", "soil_type", "elevation_band", "aspect_dir",
            "slope_band", "hydro_band"]

SPECS = [ModelSpec(
    target="cover_type",
    num_cols=NUM_COLS,
    cat_cols=["wilderness", "soil_type"],
    cheap_cols=["elevation"], cheap_depth=3,
    kind="gb",
)]


def load_covertype(n_rows: int | None = None) -> pd.DataFrame:
    from sklearn.datasets import fetch_covtype
    raw = fetch_covtype(as_frame=True).frame

    df = pd.DataFrame({
        "elevation": raw["Elevation"],
        "aspect": raw["Aspect"],
        "slope": raw["Slope"],
        "hydro_dist": raw["Horizontal_Distance_To_Hydrology"],
        "road_dist": raw["Horizontal_Distance_To_Roadways"],
        "hillshade_noon": raw["Hillshade_Noon"],
        "fire_dist": raw["Horizontal_Distance_To_Fire_Points"],
        "cover_type": raw["Cover_Type"].astype(int).map(COVER),
    })

    wa = raw[[f"Wilderness_Area_{i}" for i in range(4)]].to_numpy()
    df["wilderness"] = [WILDERNESS[i] for i in wa.argmax(axis=1)]
    st = raw[[f"Soil_Type_{i}" for i in range(40)]].to_numpy()
    df["soil_type"] = [f"soil{i:02d}" for i in st.argmax(axis=1)]

    # Coarse, human-readable static descriptors for queries.
    df["elevation_band"] = pd.cut(
        df["elevation"], bins=[0, 2400, 2700, 3000, 3300, 5000],
        labels=["lowest", "low", "mid", "high", "highest"]).astype(str)
    df["aspect_dir"] = pd.cut(
        df["aspect"] % 360, bins=[-1, 45, 135, 225, 315, 360],
        labels=["N", "E", "S", "W", "N2"]).astype(str).replace({"N2": "N"})
    df["slope_band"] = pd.cut(df["slope"], bins=[-1, 8, 16, 24, 90],
                              labels=["flat", "gentle", "steep", "extreme"]).astype(str)
    df["hydro_band"] = pd.cut(df["hydro_dist"], bins=[-1, 60, 200, 400, 10_000],
                              labels=["riparian", "near", "mid", "far"]).astype(str)

    if n_rows is not None and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=SEED)
    df = df.reset_index(drop=True)
    df["row_id"] = range(len(df))
    return df


def build_lake(min_rows: int = 300, n_rows: int | None = None,
               train_frac: float = 0.3):
    """(training_df, {source: df}, truth); predicted attribute STRIPPED."""
    df = load_covertype(n_rows=n_rows)

    rng = np.random.default_rng(SEED)
    train_idx: list[int] = []
    for _, g in df.groupby(["wilderness", "cover_type"], observed=True):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        train_idx.extend(idx[: int(len(idx) * train_frac)].tolist())
    mask = np.zeros(len(df), dtype=bool)
    mask[train_idx] = True
    df_train = df.loc[mask].reset_index(drop=True)
    df_lake = df.loc[~mask].reset_index(drop=True)

    truth = {int(r): {"cover_type": v}
             for r, v in zip(df_lake["row_id"], df_lake["cover_type"])}

    sources = {}
    for (w, soil), sub in df_lake.groupby(["wilderness", "soil_type"], observed=True):
        if len(sub) < min_rows:
            continue
        sources[f"{w}_{soil}"] = (
            sub.drop(columns=PREDICTED_ATTRS).reset_index(drop=True))
    return df_train, sources, truth


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    df_train, sources, truth = build_lake()
    lake_rows = sum(len(s) for s in sources.values())
    print(f"training {len(df_train)} rows | {len(sources)} sources | lake {lake_rows} rows")
    sizes = sorted((len(s) for s in sources.values()), reverse=True)
    print(f"source sizes: max={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}")
    vals = pd.Series([t["cover_type"] for t in truth.values()])
    print(f"cover_type: {vals.nunique()} classes {dict(vals.value_counts().head(4))}")
    assert all("cover_type" not in s.columns for s in sources.values())
    print("OK: predicted attribute stripped")
