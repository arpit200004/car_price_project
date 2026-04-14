# 🤖 Car Price Prediction using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA4F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning model that predicts the selling price of used cars based on features like company, model year, kilometers driven, fuel type, transmission, and seller type. Trained and evaluated on the CarDekho dataset.

## 📊 Model Performance Results

Below is a comparison of the different regression models tested during the development process. The **Random Forest Regressor** was selected as the final model due to its superior performance and robustness.

| Model | R² Score (Train) | R² Score (Test) | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.985** | **0.959** | **73,892** | **125,231** |
| Gradient Boosting | 0.972 | 0.956 | 76,021 | 129,383 |
| Extra Trees | 0.980 | 0.956 | 75,545 | 130,241 |
| Linear Regression | 0.882 | 0.851 | 142,500 | 210,300 |

> [!IMPORTANT]
> **Random Forest Regressor** achieved the highest accuracy with an R² score of ~96% on the test set, making it the most reliable model for this dataset.

## 🤖 Why Random Forest Regressor?

The **Random Forest Regressor** was chosen over Linear Regression and other single-model approaches for several critical reasons:

- **Handles Non-linear Relationships**: Used car pricing often follows complex, non-linear patterns that simple linear models fail to capture.
- **Robust to Outliers**: Used car data is inherently noisy; Random Forest is significantly less sensitive to outliers than models like Linear Regression.
- **Implicit Feature Selection**: The algorithm automatically identifies the most important features, reducing the need for manual selection.
- **Reduced Overfitting**: By averaging multiple decision trees, it effectively minimizes the risk of overfitting the training data.
- **Feature Importance Scores**: It provides direct insight into which factors (like car age or engine power) most significantly impact the price.
- **Accuracy**: It consistently achieves high accuracy with minimal hyperparameter tuning compared to other complex models.

## ✅ Advantages of Random Forest for this Project

- **Handles Mixed Data Types**: Effectively manages a mix of numerical features (kilometers driven) and categorical features (fuel type, transmission) after encoding.
- **No Scaling Required**: Unlike algorithms like SVM or KNN, Random Forest does not require standard feature scaling or normalization.
- **High Accuracy**: Delivers a high R² score, providing reliable estimates for a wide range of car types.
- **Interpretability via Feature Importance**: Offers clear visualization of which features most strongly drive the car's resale value.
- **Resistant to Overfitting**: The use of bagging and ensemble averaging ensures the model generalizes well to new, unseen listings.

## 🔥 Correlation Heatmap

To understand the relationships between various car features and the final selling price, a pairwise correlation analysis was performed.

- **What it shows**: Pairwise correlation coefficients between numerical features such as manufacturing year, kilometers driven, and power.
- **Key findings**:
  - `present_price` (or equivalent current market value) has the **STRONGEST positive correlation** with the selling price.
  - `year` (car age) shows a moderate positive correlation (newer cars command higher prices).
  - `km_driven` has a weak negative correlation, as higher mileage typically indicates more wear.
  - No severe multicollinearity was detected between independent features, ensuring stable model training.
- **Why it matters**: This analysis guided the selection of key drivers and confirmed which variables were redundant or most critical for accuracy.

![Correlation Heatmap](images/correlation_heatmap.png)

## 🚀 Key Features

- **Data Preprocessing**: Comprehensive handling of missing values, categorical encoding, and duplicate removal.
- **Exploratory Data Analysis**: Deep insights using distributions, boxplots, countplots, and correlation matrices.
- **Model Comparison**: Rigorous testing across Linear Regression, Decision Trees, and Random Forest.
- **Random Forest Regressor**: Implementation of the final ensemble model for the best bias-variance tradeoff.
- **Feature Importance Plot**: Visual representation of the top contributors to used car pricing.
- **Jupyter Notebook**: A complete, documented walkthrough of the entire data science lifecycle.

## 📁 Repository Structure

```text
car_price_project/
├── car_price_prediction.ipynb   # Main Jupyter Notebook
├── car data.csv                  # CarDekho dataset
├── images/
│   └── correlation_heatmap.png  # Heatmap visualization
├── requirements.txt
└── README.md
```

## 🛠️ Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook car_price_prediction.ipynb
   ```

4. **Run the model**:
   Execute all cells in the notebook to reproduce the training and evaluation results.

## 📈 How the Model Works

The prediction pipeline follows a structured machine learning workflow:

1. **Data Ingestion**: Loading and exploring the raw CarDekho dataset.
2. **Cleaning**: Handling null values and identifying statistical outliers.
3. **Encoding**: Transforming categorical features (Fuel, Transmission, Seller) into numerical formats.
4. **Analysis**: Generating a correlation heatmap to identify key price drivers.
5. **Splitting**: Dividing the processed data into training (80%) and testing (20%) sets.
6. **Training**: Fitting multiple regression models to the training data.
7. **Selection**: Identifying Random Forest as the optimal model based on evaluation metrics.
8. **Evaluation**: Final validation using R², MAE, and RMSE benchmarks.

```text
Input features → Preprocessing → Random Forest → Predicted Price (₹)
```

## ⚠️ Disclaimer

This project is intended for educational purposes. Car price predictions are estimates based on historical data and do not constitute guaranteed market valuations.

## 🙏 Credits

- Dataset provided by **CarDekho via Kaggle**.
- Core libraries: **scikit-learn**, **pandas**, **seaborn**, **matplotlib**.

---
⭐ If you find this project useful, please star the repository!
