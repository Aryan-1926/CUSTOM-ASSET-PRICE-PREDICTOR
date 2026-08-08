# Custom Asset Price Predictor (Boston Housing)

A baseline Machine Learning project implementing Ordinary Least Squares (OLS) Linear Regression to predict median asset/housing values using Scikit-Learn and Pandas.

---

## 📌 Project Overview

This repository contains a modular Python script that loads, cleans, and splits housing feature data to train a linear model estimating median home values (`medv`). 

### Data Pipeline & Cleaning Steps:
* **Missing Value Handling:** Dropped `NaN` values to ensure matrix stability during regression.
* **Filter Criteria:** Excluded invalid target entries (`medv > 0`).
* **Deduplication:** Removed duplicate records to prevent data leakage and bias.
* **Evaluation Metrics:** Scored using $R^2$ (coefficient of determination) and Mean Absolute Error (MAE) converted into real dollar amounts ($1,000s scaling).

---

## 📊 Model Performance

| Metric | Score / Value |
| :--- | :--- |
| **$R^2$ Score** | `0.6687` (~66.9% variance explained) |
| **Mean Absolute Error (MAE)** | `$3,189.09` |

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Processing:** Pandas
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `train_test_split`, `r2_score`, `mean_absolute_error`)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/Aryan-1926/CUSTOM-ASSET-PRICE-PREDICTOR.git](https://github.com/Aryan-1926/CUSTOM-ASSET-PRICE-PREDICTOR.git)
cd CUSTOM-ASSET-PRICE-PREDICTOR
