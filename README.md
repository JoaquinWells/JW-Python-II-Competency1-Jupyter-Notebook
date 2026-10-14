# Competency 1: First Machine Learning System

## Project Purpose

This project builds a machine-learning system that estimates a property's house price per unit area. The estimate is intended to support a real estate agent's initial review. It is not a formal appraisal.

## Dataset

The project uses the UCI Real Estate Valuation dataset. The model predicts `Y house price of unit area` from these six features:

- Transaction date
- House age
- Distance to the nearest MRT station
- Number of nearby convenience stores
- Latitude
- Longitude

The dataset's `No` column was not used because it only identifies each record and has no meaningful predictive value. The transaction date was kept as a numeric value because its decimals represent the timing of the transaction.

The dataset contains 414 records. During exploration, numeric input features were found, with no missing values, and no obviously impossible ranges. Scatterplots and correlations were used to look for relationships between each feature and the target. Distance to the nearest MRT station had the strongest relationship in this dataset, but correlation does not prove causation.

The target is the property's price per ping. One ping is approximately 3.306 square meters, and the dataset reports prices in units of 10,000 New Taiwan dollars per ping.

## Workflow

The data was divided into training and test sets before modeling. 80% percent was used for training and 20% was held back for testing. A fixed random state makes the split reproducible.

The model uses a scikit-learn pipeline. `StandardScaler` puts the numeric features on comparable scales, and `LinearRegression` estimates the target. The scaler is fit only with the training data so information from the test data does not influence the model.

The model was evaluated using properties it did not see during training:

- **Mean absolute error:** 5.305 target units. Predictions were about 5.3 units higher or lower than the actual values on average.
- **Mean squared error:** 53.506. This gives more weight to larger prediction errors.
- **R-squared:** 0.68. The model explained about 68% of the variation in the test prices. This is not the same as being 68% accurate.

The final scatterplot compares actual and predicted test values. Points closer to the diagonal reference line represent more accurate predictions.

## Example Predictions

The three example estimates are reported in the dataset's target unit: 10,000 New Taiwan dollars per ping.

| Property | Estimated price/unit area (10,000 NTD/ping) |
| --- | ---: |
| Property A | 46.91 |
| Property B | 35.66 |
| Property C | 50.05 |

These are estimates per unit area, not total property prices or price changes.

## Limitations

Testing only on the training data can create a misleading impression because the model has already seen those examples and may appear better than it performs on new properties. The held-back test set provides a more realistic performance estimate.

This dataset represents transactions from one location and time period. The model may not generalize to another market or a different time. It also does not include details such as property condition, renovations, floor level, building quality, or current market conditions. Agents should use the estimates as a starting point and complete a detailed market analysis before making decisions.

## Setup

Create and activate a virtual environment, then install the dependencies in the terminal:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Open `competency1.ipynb` in VS Code and select the interpreter from `venv/bin/python`. The notebook downloads the dataset automatically through `ucimlrepo`, so no separate dataset download is needed. An internet connection is required when the notebook loads the data.

The notebook also includes a small macOS certificate setup so Python can connect securely to the UCI website. This setup is only needed for the dataset download on Mac and does not change the data or model.

```python
dataset = fetch_ucirepo(id=477)
```

## Sources

Yeh, I. (2018). *Real Estate Valuation* [Dataset]. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5J30W). Licensed under CC BY 4.0.

Scikit-learn. “Preprocessing data.” Scikit-learn documentation. https://scikit-learn.org/stable/modules/preprocessing.html