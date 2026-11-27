---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - sklearn
  - tabular-regression
  - real-estate
  - house-prices
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - mae
  - rmse
  - r2
  - mape
---

# King County house price model: predicting sale price from a house's sales sheet

This model predicts the **sale price in US dollars** of a residential house in King County, Washington (Seattle area). It uses 18 facts found on a sales sheet: living area, lot size, rooms, floors, build quality grade, condition, view, waterfront, year built, renovation, zipcode, latitude and longitude, and the size of the neighbouring houses. It is meant for **real estate agents and small investors** who want a quick value estimate to compare with an asking price. It is a scikit-learn pipeline (preprocessing + `HistGradientBoostingRegressor` trained on log(price)), fitted on about 17,000 house sales from 2014–2015. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/group7_day2_assignment
- **Team:** Ugochinyere Megwa, Sneha Sachin Shenoy, Álvaro Miñé

## Intended use

- **Intended users and use:** real estate agents, small investors and students who want a first estimate of what a King County house is worth, to compare it with the asking price, or to compare two options (for example a house as it is versus the same house upgraded). The estimate supports a human decision; it does not replace a professional appraisal.
- **Out-of-scope uses:**
  - **Other regions or countries:** prices, zipcodes and the `grade` scale are specific to King County, USA.
  - **Today's absolute prices:** the data covers May 2014 – May 2015. Use the model for *comparisons*, not as a current market value.
  - **Other building types:** apartments, commercial buildings and empty land are not in the data.
  - **Luxury and waterfront homes:** few examples; predictions above about $1M or for waterfront houses are systematically too low (treat them as a lower bound).
  - **Decisions about people:** lending, insurance, tax assessment or anything that affects individuals. The model reproduces past market prices, including historical differences between neighbourhoods.

## Model details

| | |
|---|---|
| **Problem type** | Regression (target: `price`, US dollars) |
| **Algorithm** | `HistGradientBoostingRegressor`, wrapped in a `TransformedTargetRegressor` (`func=np.log`, `inverse_func=np.exp`) |
| **Key hyperparameters** | `learning_rate=0.05`, `max_iter=800`, `max_leaf_nodes=31`, early stopping (scikit-learn default `early_stopping="auto"`, active above 10,000 rows), `random_state=42` |
| **Preprocessing** | `ColumnTransformer`: numeric → median imputation + `StandardScaler`; ordinal (`grade`, `condition`, `view`) → most-frequent imputation + `OrdinalEncoder` with an explicit order + `StandardScaler`; nominal (`zipcode`) → most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`; binary flags → passed through |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` (preprocessing + model), 1.1 MB, saved with `joblib.dump` |
| **Library versions** | Python 3.11, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, joblib 1.6.0 (see `requirements.txt` in the GitHub repository) |
| **Authors** | Ugochinyere Megwa, Sneha Sachin Shenoy, Álvaro Miñé |

## Input features

The model expects a `pandas.DataFrame` with exactly these 18 columns (column names must match; order as below is safest):

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `bedrooms` | numeric | count | Number of bedrooms | median imputation, scaling |
| `bathrooms` | numeric | count | Number of bathrooms (0.25 steps: 0.5 = toilet only, 0.75 = shower, 1 = full) | median imputation, scaling |
| `sqft_living` | numeric | sq ft | Interior living area | median imputation, scaling |
| `sqft_lot` | numeric | sq ft | Lot (land) area | median imputation, scaling |
| `floors` | numeric | count | Number of floors (0.5 steps) | median imputation, scaling |
| `sqft_basement` | numeric | sq ft | Basement area (0 = no basement) | median imputation, scaling |
| `yr_built` | numeric | year | Year of construction | median imputation, scaling |
| `lat` | numeric | degrees | Latitude | median imputation, scaling |
| `long` | numeric | degrees | Longitude | median imputation, scaling |
| `sqft_living15` | numeric | sq ft | Average living area of the 15 nearest houses | median imputation, scaling |
| `sqft_lot15` | numeric | sq ft | Average lot area of the 15 nearest houses | median imputation, scaling |
| `size_vs_neighbours` | numeric | ratio | Built by us: `sqft_living / sqft_living15` (> 1 = bigger than the neighbours) | median imputation, scaling |
| `grade` | ordinal (int) | 1–13 | King County assessor's build quality grade (1 = poor … 7 = average … 13 = mansion quality) | `OrdinalEncoder` (order 1…13), scaling |
| `condition` | ordinal (int) | 1–5 | State of maintenance (1 = poor … 5 = very good) | `OrdinalEncoder` (order 1…5), scaling |
| `view` | ordinal (int) | 0–4 | Quality of the view (0 = none … 4 = excellent) | `OrdinalEncoder` (order 0…4), scaling |
| `zipcode` | nominal (string) | – | 5-digit zipcode, as text (e.g. `"98033"`) | one-hot encoding (unknown zipcodes ignored) |
| `waterfront` | binary (0/1) | – | 1 if the house is on the waterfront | passed through |
| `renovated` | binary (0/1) | – | Built by us: 1 if the house was ever renovated | passed through |

Removed on purpose: `id` (identifier), `date` (all sales from one year), `sqft_above` (exactly `sqft_living − sqft_basement`), `yr_renovated` (replaced by `renovated`), and anything computed from the price (leakage).

## Output

`model.predict(X)` returns one number per row: the **estimated sale price in US dollars** (2014–2015 prices). The model is trained on log(price), but the `TransformedTargetRegressor` converts predictions back automatically, so **no conversion is needed** on the user side. Because of the log transform, the prediction is closer to a median than a mean price for similar houses.

## Training data

| | |
|---|---|
| **Dataset** | Housing Price Dataset (house sales in King County, Washington, USA) |
| **Source** | [Kaggle – sukhmandeepsinghbrar/housing-price-dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset) |
| **Licence** | To be confirmed on the Kaggle page. The data comes from King County public sales records and is also published on Kaggle as "House Sales in King County, USA" under CC0 (public domain). |
| **One row is** | One house sale: price, sizes, rooms, floors, grade, condition, view, waterfront, year built / renovated, zipcode, coordinates, neighbours' average sizes |
| **Size used** | 21,418 houses (train 17,134; test 4,284) out of 21,613 raw rows × 21 columns |
| **Coverage** | King County, WA, USA; sales from May 2014 to May 2015; residential houses only |

**Cleaning:** 18 impossible rows were removed (0 bedrooms, 0 bathrooms, or more than 10 bedrooms), and for the 177 houses sold twice in the period only the most recent sale was kept, so the same house cannot appear in both train and test. Two features were built: `renovated` (from `yr_renovated`) and `size_vs_neighbours`. Details are in `notebook/data_cleaning_and_EDA.ipynb` in the GitHub repository. (Our Day 1 dataset, NYC renovation permits, was replaced because it did not contain enough usable data for prediction.)

## Training procedure

- **Split:** 80 / 20 train/test (17,134 / 4,284 houses), `random_state=42`, not stratified (regression). The test set was used **once**, at the end.
- **Validation:** 5-fold `KFold` cross-validation (shuffled, `random_state=42`) on the training set, same folds and metrics for every model.
- **Target:** all models learn log(price) (price skewness 4.0 → 0.4 after the log); metrics are computed in dollars.
- **Models compared** (5-fold CV, mean ± std):

| Model | MAE ($) | RMSE ($) | R² | MAPE (%) |
|---|---|---|---|---|
| Baseline (`DummyRegressor`, median) | 222,072 ± 4,184 | 378,509 ± 24,063 | −0.061 ± 0.008 | 42.2 ± 0.7 |
| Ridge (`alpha=1.0`) | 74,964 ± 1,692 | 137,588 ± 18,331 | 0.860 ± 0.024 | 13.7 ± 0.2 |
| KNN (k = 10, distance weights) | 84,965 ± 2,337 | 175,190 ± 18,180 | 0.774 ± 0.019 | 14.6 ± 0.3 |
| Random forest (200 trees) | 70,368 ± 1,399 | 135,381 ± 6,194 | 0.864 ± 0.009 | 12.6 ± 0.1 |
| Gradient boosting (default) | 65,367 ± 1,846 | 123,631 ± 13,318 | 0.887 ± 0.014 | 12.0 ± 0.1 |
| **Gradient boosting, tuned** | **64,011 ± 1,553** | | | |

- **Tuning:** a validation curve of `max_depth`, then `GridSearchCV` (5-fold, scoring MAE) over `learning_rate` ∈ {0.05, 0.1}, `max_iter` ∈ {300, 800}, `max_leaf_nodes` ∈ {15, 31, 63}. Best: `learning_rate=0.05`, `max_iter=800`, `max_leaf_nodes=31`. The gain over the default (−$1.4K MAE) is within noise, but costs nothing.
- **Why this model:** gradient boosting has the lowest error on every metric. It beats the random forest by about $5K MAE (about 3 standard deviations) and trains about 10× faster. The final pipeline was refitted on the whole training set by `GridSearchCV`.

## Evaluation

Results on the held-out **test set** (4,284 houses, used once):

| Metric | Test | Cross-validation (train) | Baseline (CV) |
|---|---|---|---|
| MAE | **$62,185** | $64,011 ± $1,553 | $222,072 |
| RMSE | $108,202 | – | $378,509 |
| R² | 0.911 | – | −0.061 |
| MAPE | 11.6 % | – | 42.2 % |
| Median absolute error | $37,085 | – | – |

The test MAE agrees with the cross-validation MAE, so the model is not over-fitted to the training folds.

**Where it performs worst** (median signed error on the test set; negative = under-estimated):

| Group | Median error |
|---|---|
| Houses over $1M | −7.5 % (under-estimated) |
| Waterfront houses | −12.8 % (under-estimated) |
| Houses under $300K | +3.7 % (over-estimated) |

The model pulls extreme houses towards the middle of the market. The costly mistake for our user, under-estimating a good property, is concentrated on luxury and waterfront homes.

**What it relies on** (permutation importance on the test set): location first (shuffling `lat` raises the MAE by about $110K), then living area and build quality (`grade`). `bedrooms` adds almost nothing once the living area is known, and `condition` matters much less than `grade`.

## Limitations and biases

- **Time:** sales from May 2014 – May 2015 only. Prices are outdated; use predictions for comparisons (option A vs B), not as today's market value.
- **Place:** King County, USA only. Zipcodes outside the training data are ignored by the encoder, so predictions for them rely on coordinates alone and are unreliable. The model does not transfer to other regions or countries.
- **Building types:** residential houses only; not apartments, commercial buildings or empty land. `grade` is King County's own assessor scale and must be supplied on that scale.
- **Value range:** few luxury and waterfront examples, so these houses are under-estimated (−7.5 % and −12.8 % median). Treat predictions above about $1M, or for waterfront houses, as a lower bound. Cheap houses (< $300K) are slightly over-estimated.
- **Evaluation assumption:** the train/test split is random, so the same neighbourhoods appear in both. The scores measure performance within known areas and the same year, not in a new area or a future market.
- **Leakage:** columns computed from the price were removed; houses sold twice were de-duplicated so no house is in both train and test. The neighbours' sizes (`sqft_living15`, `sqft_lot15`) come from the dataset and are assumed known at prediction time.
- **Bias:** the model learns past market prices, including historical price differences between neighbourhoods. It describes correlations, not causes (for example, the effect of an upgrade is an estimate, not a guarantee), and must not be used for decisions about people (lending, insurance).

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day2-team7-house-price", filename="model.joblib")
model = joblib.load(path)

house = {
    "bedrooms": 3, "bathrooms": 2.25, "sqft_living": 2000, "sqft_lot": 7500, "floors": 2.0,
    "sqft_basement": 0, "yr_built": 1992, "lat": 47.68, "long": -122.19,
    "sqft_living15": 2100, "sqft_lot15": 7500,
    "grade": 7, "condition": 3, "view": 0, "zipcode": "98033",
    "waterfront": 0, "renovated": 0,
}
house["size_vs_neighbours"] = house["sqft_living"] / house["sqft_living15"]

columns = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_basement",
           "yr_built", "lat", "long", "sqft_living15", "sqft_lot15", "size_vs_neighbours",
           "grade", "condition", "view", "zipcode", "waterfront", "renovated"]
example = pd.DataFrame([house])[columns]

print(f"Estimated price: ${model.predict(example)[0]:,.0f}")   # about $659,000
```

## Security note

⚠️ `.joblib` files are Python pickles: **loading one can execute arbitrary code**. Only load this model (or any `.joblib` file) from a source you trust, such as this repository, and use the library versions listed above; the model may not load, or may behave differently, with other versions of scikit-learn.

## Reproducibility

The full pipeline (data loading, cleaning, EDA, training, evaluation, interpretation) is in the GitHub repository: https://github.com/iaac-maai/group7_day2_assignment.

1. `pip install -r requirements.txt` (pinned versions: scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, joblib 1.6.0, huggingface_hub 2.3.0).
2. Run `notebook/data_cleaning_and_EDA.ipynb` (creates `data/housing_clean.csv`).
3. Run `notebook/supervised_learning.ipynb` (about 5 minutes; exports the figures and `models/model.joblib`).

All random steps use `random_state=42`.

## Authors and licence

Ugochinyere Megwa, Sneha Sachin Shenoy, Álvaro Miñé. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model and code released under the MIT License. The training data is subject to its own licence: King County public sales records, published on Kaggle ([sukhmandeepsinghbrar/housing-price-dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset); the same records are also on Kaggle as "House Sales in King County, USA" under CC0).
