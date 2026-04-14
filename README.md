# DriveValue 🚘

**DriveValue** is a premium, ML-powered used car valuation engine tailored specifically for the Indian automotive market. It fuses high-precision statistical learning with a stunning, consumer-grade **Champagne Gold 3D Glass** interface to deliver accurate real-time pricing intelligence.

![DriveValue Interface](https://img.shields.io/badge/UI-Champagne%20Gold%203D%20Glass-cda434?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white) 
![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🔥 Key Features

- **Precision Valuations**: Built on a `HistGradientBoostingRegressor` pipeline that accounts for deep market nuances, such as accurate physical car age depreciation and 2024 economic inflation adjustments.
- **Micro-Variant Awareness**: Distinguishes between hyper-specific car variants (e.g. *Honda Accord 2.4 MT* vs *Honda Accord Hybrid*) to completely eliminate false pricing anomalies.
- **Consumer-Grade Aesthetics**: A meticulous "Champagne Gold" aesthetic heavily employing custom CSS tokens, ambient glows, dynamic popups, and modern frontend 3D-animations—all without an external UI framework.
- **Institutional Market Band**: Returns not only a direct price but compares against the current equivalent market percentiles and dealer / individual discount realities.
- **Immersive Loader**: Custom CSS key-framed "revving car" animations ensure a high-end interaction paradigm.

## 🛠 Tech Stack

- **Machine Learning**: `scikit-learn` (ExtraTrees, HistGradientBoosting Regressors)
- **Data Engineering**: `pandas`, `numpy`
- **Frontend / Application**: `streamlit` & Custom CSS Injections.

## 📂 Project Structure

```text
├── app.py                # Main Application & Frontend (Streamlit + CSS + Interactions)
├── train_model.py        # ML Training Pipeline & Execution (Generates model bundles)
├── model_data.py         # Advanced Data Engineering (Inflation mapping, Age modeling)
├── data/                 # Raw & Synthetic augmentation datasets (Ignored in Git)
├── model_report.json     # Dynamic JSON report generated per model training instance
└── car_price_model_v3.pkl# Compiled Binary Model (Ignored in Git)
```

## 🚀 Getting Started

### 1. Requirements & Installation
Ensure you have Python 3.9+ and set up your virtual environment via Anaconda or `venv`.

```bash
# Clone the repository
git clone https://github.com/your-username/drivevalue.git
cd drivevalue

# Install core required packages
pip install streamlit pandas numpy scikit-learn
```

### 2. Model Training (Optional)
If you want to recompute the latest inflation thresholds and variant mappings, run the training pipeline directly. This will construct the `.pkl` bundle and spit out a performance correlation matrix.

```bash
python train_model.py
```

### 3. Running the Interface
Fire up the interface engine sequentially by running:

```bash
streamlit run app.py
```

## 🔮 Engineering Notes 

**The Data Depreciation Gap**: Unlike many academic Kaggle attempts that blindly calculate `current_year - model_year`, DriveValue implements a split chronological strategy. It calculates the **true physical age of the car at the time of the transaction recording** to correctly teach the model physical item depreciation curves, and then normalizes those isolated historical targets securely to *current-year economic valuations* simulating an active post-COVID ~5.5% used car inflation trend. This achieves a state-of-the-art accuracy that raw uncalibrated scraping cannot natively provide.

---

> Designed & Engineered for the modern automotive economy.
