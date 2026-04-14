# DriveValue: Advanced Indian Context Car Price Estimator

DriveValue is an enterprise-grade, machine-learning-powered used car valuation engine specifically engineered for the Indian automotive market. Built utilizing advanced statistical learning methodologies, the system couples high-precision market modeling with an aesthetic Dark Champagne Gold 3D Glass UI.

## 1. System Output and Objectives

The primary objective of DriveValue is to eliminate standard machine learning hallucination phenomena by precisely mapping market depreciation curves and distinguishing between hyper-specific car variants. DriveValue identifies nuanced discrepancies (e.g., Honda Accord 2.4 MT vs. Honda Accord Hybrid) and provides real-time pricing intelligence based on current economic valuations. 

## 2. Dataset Architecture

The system utilizes an aggregated matrix drawn primarily from the [CarDekho Used Cars Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), supplemented with synthetic modern 2024 augmentation vectors to maintain chronological relevance.

### Core Metrics:
* Observations: ~35,000 viable listings after sanitization
* Features: ~20 contextual, physical, and engineering-related datapoints
* Coverage base: Across multiple major Indian tier-1 and tier-2 cities

### Price Calibration:
Because dealer listings inherently possess markup margins designed for negotiation, the base dataset represents "Asking Prices". Our preprocessing pipeline executes markdown coefficients mapping them directly to True Transaction Prices:
* Dealer listings receive a static 15% markdown.
* Individual listings receive a static 10% markdown.
* Prices that end exactly on a clean multi-thousand multiple (e.g. INR 500,000) are programmatically identified as "aspirational" and receive an additional 2% reduction.

## 3. Preprocessing Pipeline

The primary model data orchestrator aggressively cleanses the data prior to ingestion:

* Dimension Consistency: Missing engine displacements and physical dimensions (Length/Width/Height) were iteratively inferred utilizing cluster median assignments via the primary chassis and model classification.
* Statistical Hard-Bounds: Implementation of strict physiological thresholds:
  * Prices below INR 60,000 and above INR 12,000,000 excluded.
  * Engine capabilities scaled between 500cc and 6500cc boundaries.
  * Implausible mileages and outlier-driven records explicitly eradicated via 99.5th percentile truncations.

## 4. Substantive Feature Engineering

The system pivots away from raw calendar year data to understand physical aging and true mechanical depreciation.

* Listing Contextualization: Extracting a static `2024 - model_year` inherently inflates valuations by evaluating historic transaction values with future age properties. DriveValue interprets the true `listing_year` (approximated as 2021 for the primary Kaggle node) and calculates `car_age = listing_year - model_year`. 
* Inflation Normalization: The system can adjust isolated historic transactions into current-year equivalents, preventing extreme chronological price-banding errors.
* Usage Severity Metrics: `km_per_year` is dynamically generated during the prediction sequence to distinguish heavily abused commercial vehicles from light-use individual commuting profiles.

## 5. Model Selection and Categorical Encoding

We utilize a structured `HistGradientBoostingRegressor` to accommodate extensive tabular structures containing categorical dimensionality. It actively outperforms baseline GradientBoosting and ExtraTrees in variance mapping while processing inference queries in fractions of a millisecond.

### Encoding Strategy
Noticeably, we bypassed standard One-Hot Encoding (OHE). Given the highly scattered sparsity of Indian car configurations (over 800+ independent variants), OHE creates an extreme memory cost often triggering the Curse of Dimensionality.

Instead, DriveValue utilizes `sklearn.preprocessing.OrdinalEncoder`. Histograms and decision trees natively calculate optimal categorical splits without requiring bloated binary matrix expansion. Out-of-vocabulary anomalies triggered during future interactions are explicitly routed to a safe fallback via `-1` encoded integers.

## 6. Execution and Operation

Ensure you are operating within a structured virtual-environment containing `streamlit`, `pandas`, `numpy`, and `scikit-learn`.

To invoke the primary application interface:
```bash
python -m streamlit run app.py
```

To reconstruct the model parameters and pipeline structures:
```bash
python train_model.py
```
