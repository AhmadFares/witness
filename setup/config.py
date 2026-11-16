"""Schema + partitioning constants."""

STATIC_ATTRS = [
    "age", "workclass", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
]
PREDICTED_ATTR = "income"           # values: "<=50K" / ">50K"
POSITIVE_LABEL = ">50K"
PARTITION_KEY = "native_country"

REGION_MAP = {
    "United-States": "North_America", "Canada": "North_America",
    "Outlying-US(Guam-USVI-etc)": "North_America",
    "Mexico": "Latin_America", "Puerto-Rico": "Latin_America",
    "Cuba": "Latin_America", "Jamaica": "Latin_America",
    "Dominican-Republic": "Latin_America", "Haiti": "Latin_America",
    "Guatemala": "Latin_America", "Honduras": "Latin_America",
    "Nicaragua": "Latin_America", "El-Salvador": "Latin_America",
    "Columbia": "Latin_America", "Ecuador": "Latin_America",
    "Peru": "Latin_America", "Trinadad&Tobago": "Latin_America",
    "England": "Europe", "Germany": "Europe", "Italy": "Europe",
    "Poland": "Europe", "Portugal": "Europe", "France": "Europe",
    "Greece": "Europe", "Ireland": "Europe", "Scotland": "Europe",
    "Yugoslavia": "Europe", "Hungary": "Europe", "Holand-Netherlands": "Europe",
    "Philippines": "Asia", "India": "Asia", "China": "Asia",
    "Japan": "Asia", "Vietnam": "Asia", "Taiwan": "Asia",
    "Cambodia": "Asia", "Thailand": "Asia", "Laos": "Asia",
    "Hong": "Asia", "South": "Asia",
    "Iran": "Middle_East",
}

SEED = 42
