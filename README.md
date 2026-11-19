# Witness

Witness finds compact *witness tables* in data lakes of horizontally partitioned
tables, for queries with **stored conditions** (answered by the tables) and
**predicted conditions** (answered by an expensive model at query time).
It supports two semantics:

- **Associative**: the requested values co-occur in as few tuples as possible
  (target: *AWR* ≥ θ).
- **Diverse**: every requested value is covered, with as many other values as
  possible (targets: *TWR* ≥ θ_T and *DWR* ≥ θ_D).

Witness reads one source table at a time. It chooses each table from local
statistics and a few cheap predictions on samples, makes expensive predictions
in chunks, and stops as soon as the target is reached.

## Install

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

Run every command below from the root of the repository.

## Data

- **ACSIncome**: downloaded automatically from OpenML (id 43137) on first use,
  into `data/openml/`.
- **MovieLens 25M + IMDb**: these datasets may not be redistributed, so download
  them yourself:
  - MovieLens 25M from <https://grouplens.org/datasets/movielens/25m/>; unzip
    it into `data/ml-25m/`.
  - `title.basics.tsv.gz` and `title.ratings.tsv.gz` from
    <https://datasets.imdbws.com/> into `data/imdb/`.

The expensive model *M* (gradient boosting) and the cheap model *M_c* (a
depth-2 decision tree) are trained on first use and saved in
`demo/saved_models/`.

## Python API

```python
import witness

# Load the data lake and its models
lake = witness.load_datalake("ACSIncome", split="state x education")
M = witness.load_model(lake)          # expensive model
M_c = witness.load_cheap_model(lake)  # cheap model

# Associative query on stored and predicted attributes
query = {"sex": ["Female"],
         "education": ["No high school diploma", "High school"],
         "income": [">50K"]}
T, stats = witness.query(
    lake, query, semantics="associative",
    target_threshold=0.8, model=M, cheap_model=M_c)
print(f"Retained {len(T)} tuples; expensive predictions: {stats.m_calls}")
```

`stats` also gives the cheap predictions, the tables scanned, the table
accesses and the measures (*AWR*, *TWR*, *DWR*).

## Demo

```bash
streamlit run demo/app.py
```

The dashboard has three parts:

- **Ask**: choose a dataset and its split, then write stored and predicted
  conditions or start from an example question.
- **Answer**: the witness table and its measures, with a comparison against a
  single SQL query on 5 small tables.
- **How It Was Found**: Witness against Cheap Proxy, Random Order, Exhaustive
  Prediction and three ablations, with costs computed from editable prices.

## Repository layout

| Folder | Contents |
|---|---|
| `witness/` | the Python API |
| `pipeline.py`, `boxes/`, `core/` | the search: source selection, chunked prediction, pruning, measures |
| `setup/` | building the data lakes and the models |
| `demo/` | the Streamlit dashboard |
| `experiments/` | query generation and baseline helpers used by the demo |

## Paper

Ahmad Fares, Silviu Maniu, Sihem Amer-Yahia. *Witness: Finding Evidence in
Tables with Stored and Predicted Attributes.* Submitted to the EDBT 2027
demonstration track.
