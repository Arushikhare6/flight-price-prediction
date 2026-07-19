# Flight Price Prediction

Predicting airline ticket prices using regression modeling, hyperparameter tuning, and SHAP-based explainability, served through an interactive Streamlit application.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Regression-yellow)
![CatBoost](https://img.shields.io/badge/CatBoost-Regression-lightgrey)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Architecture / Workflow](#project-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Project Overview

Airline ticket prices fluctuate based on a combination of factors — airline, route, number of stops, journey timing, and duration — that are not transparent to the end buyer and are non-trivial to model directly.

This project builds a complete regression pipeline that predicts flight ticket prices from these underlying factors, benchmarks multiple algorithms against each other, tunes the strongest candidates, and explains individual predictions using SHAP rather than treating the model as a black box.

**Why it matters:**
- Buyers can get a data-driven price estimate before booking, rather than relying on guesswork or manual comparison across listings.
- Travel platforms can use similar pipelines to power price-alert features, fare-trend analysis, or anomaly detection on mispriced listings.
- The explainability layer means predictions are auditable — a stakeholder can see *why* a given price was predicted, not just the number itself.

**Target users:** Travel-tech platforms, fare comparison tools, and data teams evaluating fare prediction as a feature; also serves as a reference implementation of a leakage-safe, explainable regression pipeline.

---

## Features

- End-to-end pipeline from raw data to deployed prediction app
- Benchmarking across 8 regression algorithms under identical, leakage-safe conditions
- Automated hyperparameter tuning via `RandomizedSearchCV` on top-performing candidates
- Model explainability with SHAP (global feature importance and per-prediction breakdowns)
- Modular, production-style codebase (`src/`) shared across notebooks and the deployed app
- Interactive Streamlit application for live, explainable predictions
- Reproducible environment via a pinned `requirements.txt`

---

## Dataset

| Attribute | Details |
|---|---|
| **Source** | Kaggle — Flight Price Prediction dataset |
| **Size** | ~10,000+ flight records |
| **Target Variable** | `Price` (ticket price) |
| **Key Features** | Airline, Source, Destination, Route, Date of Journey, Departure Time, Arrival Time, Duration, Total Stops, Additional Info |

**Preprocessing performed:**
- Parsed `Date_of_Journey` into `Journey_Day`, `Journey_Month`, and `Journey_Weekday`
- Converted departure/arrival timestamps into separate hour and minute features
- Converted flight duration from string format (e.g., `2h 50m`) into total minutes
- Converted `Total_Stops` from text categories (e.g., `non-stop`, `1 stop`) into ordinal integers
- Dropped low-information columns (`Additional_Info`, `Route`) after evaluation
- Removed missing values and duplicate records
- Encoded categorical variables (`Airline`, `Source`, `Destination`) using One-Hot Encoding within a `ColumnTransformer`, wrapped in a `Pipeline` to prevent train/test data leakage

---

## Tech Stack

| Category | Tools |
|---|---|
| **Programming Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, Joblib |
| **Machine Learning Models** | Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, XGBoost, LightGBM, CatBoost |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Plotly |
| **Deployment** | Streamlit |
| **Version Control** | Git, GitHub |

---

## Project Architecture / Workflow

<details>
<summary><strong>Click to expand full pipeline description</strong></summary>

1. **Data Collection** — Raw flight data sourced from Kaggle, stored under `data/raw/`.
2. **Data Cleaning** — Missing values, duplicates, and invalid duration entries handled and validated.
3. **Exploratory Data Analysis (EDA)** — Price distributions analyzed against airline, source, destination, stops, and duration to identify key drivers.
4. **Feature Engineering** — Date and time fields decomposed into model-friendly numeric features; duration standardized to minutes.
5. **Preprocessing** — Categorical features encoded via `OneHotEncoder` inside a `ColumnTransformer`; entire preprocessing step wrapped in a `Pipeline` alongside each model to guarantee identical, leakage-free treatment of train and test data.
6. **Model Training** — Eight regression algorithms trained under identical pipeline conditions for a fair, apples-to-apples comparison.
7. **Hyperparameter Tuning** — Top three baseline performers tuned via `RandomizedSearchCV` with cross-validation.
8. **Evaluation** — Models compared using MAE, RMSE, and R² Score; tuned results validated against baseline before being accepted as final.
9. **Model Saving** — Best-performing pipeline serialized with `joblib` for reuse without retraining.
10. **Prediction / Explainability** — Final model deployed in a Streamlit app; each prediction is accompanied by a SHAP waterfall plot showing which features increased or decreased the predicted price.

</details>

---

## Repository Structure

```
flight-price-prediction/
│
├── app/
│   └── app.py                     # Streamlit prediction app
│
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # Cleaned and feature-engineered dataset
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Preprocessing_and_Model_Building.ipynb
│   ├── 06_Hyperparameter_Tuning.ipynb
│   └── 07_SHAP_Explainability.ipynb
│
├── src/
│   ├── preprocessing.py            # Preprocessing pipeline (ColumnTransformer)
│   ├── model_factory.py            # Candidate model definitions
│   ├── evaluation.py               # Regression evaluation metrics
│   └── tuning.py                   # Hyperparameter search spaces
│
├── models/
│   ├── best_baseline_model.pkl
│   └── final_tuned_model.pkl
│
├── results/
│   ├── baseline_results.csv
│   ├── tuned_results.csv
│   └── shap_feature_importance.csv
│
├── screenshots/
│   ├── baseline_r2_comparison.png
│   ├── shap_summary_plot.png
│   └── shap_waterfall_single_prediction.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Arushikhare6/flight-price-prediction.git
cd flight-price-prediction

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

**Run the notebooks (optional — to reproduce training from scratch):**

```bash
jupyter notebook notebooks/
```

Run notebooks `01` through `07` in sequence to reproduce data preparation, model training, tuning, and explainability outputs.

**Run the prediction app:**

```bash
streamlit run app/app.py
```

> If `models/final_tuned_model.pkl` is not present, regenerate it by running `notebooks/05_Preprocessing_and_Model_Building.ipynb` and `notebooks/06_Hyperparameter_Tuning.ipynb` in order.

---

## Models Used

| Model | Purpose | Advantages |
|---|---|---|
| Linear Regression | Baseline benchmark | Simple, interpretable, fast to train |
| Decision Tree | Non-linear baseline | Captures non-linear relationships, easy to visualize |
| Random Forest | Ensemble baseline | Reduces overfitting via bagging, handles non-linearity well |
| Gradient Boosting | Sequential ensemble | Corrects prior errors iteratively, strong predictive power |
| Extra Trees | Randomized ensemble | Faster training than Random Forest, reduces variance |
| XGBoost | Gradient boosting (optimized) | Regularization, high performance, efficient on tabular data |
| LightGBM | Gradient boosting (leaf-wise) | Faster training on large datasets, memory efficient |
| CatBoost | Gradient boosting (categorical-aware) | Native categorical feature handling, resistant to overfitting |

---

## Model Evaluation

Baseline and tuned models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

| Model | MAE (₹) | RMSE (₹) | R² Score |
|---|---|---|---|
| **XGBoost (Tuned)** | 1171.42 | 1819.60 | **0.8412** |
| Random Forest (Tuned) | 1129.60 | 1873.92 | 0.8316 |
| LightGBM (Tuned) | 1162.70 | 1926.26 | 0.8220 |
| Decision Tree | *[placeholder]* | *[placeholder]* | *[placeholder]* |
| Gradient Boosting | *[placeholder]* | *[placeholder]* | *[placeholder]* |
| Extra Trees | *[placeholder]* | *[placeholder]* | *[placeholder]* |
| CatBoost | *[placeholder]* | *[placeholder]* | *[placeholder]* |
| Linear Regression | *[placeholder]* | *[placeholder]* | *[placeholder]* |

---

## Results

The final tuned **XGBoost** model was selected as the production model, achieving an R² Score of **0.8412** on the held-out test set — explaining approximately 84% of the variance in flight ticket prices, with an average prediction error (MAE) of approximately ₹1,171.

SHAP analysis was used to validate that the model's predictions align with domain intuition (e.g., number of stops and airline choice are meaningful price drivers), rather than relying on spurious correlations. Full feature-level importance is available in `results/shap_feature_importance.csv`.

---

## Future Improvements

- Graceful handling of unseen airlines or routes at inference time
- Scheduled retraining pipeline to incorporate new flight data
- Incorporation of seasonal and holiday demand signals for dynamic pricing awareness
- Confidence intervals or prediction ranges alongside point estimates
- CI/CD pipeline for automated testing and deployment
- Model monitoring for prediction drift in production

---

## Screenshots

**Model Comparison**

`[Insert screenshot: screenshots/baseline_r2_comparison.png]`

**SHAP Global Feature Importance**

`[Insert screenshot: screenshots/shap_summary_plot.png]`

**SHAP Single Prediction Explanation**

`[Insert screenshot: screenshots/shap_waterfall_single_prediction.png]`

**Streamlit Application**

`[Insert screenshot: app interface and sample prediction]`

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes with clear, descriptive messages
4. Push to your branch and open a Pull Request
5. Ensure code follows existing style conventions and includes relevant documentation updates

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Name:** Arushi Khare
**GitHub:** [@Arushikhare6](https://github.com/Arushikhare6)
**LinkedIn:** *[placeholder]*
**Email:** *[placeholder]*
