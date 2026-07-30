# Jakarta Air Quality Index (AQI) Analysis & Prediction

A machine learning project built from scratch using NumPy to analyze how atmospheric pollutants impact Jakarta's daily Air Quality Index (AQI) and predict daily maximum AQI levels.

---

## 📌 Project Overview
Air pollution in urban centers like Jakarta poses severe public health risks. This project investigates the relationship between major atmospheric pollutants and Jakarta's daily AQI. 

Using data spanning from 2010 to 2021, we implemented a **Multiple Linear Regression model trained via Gradient Descent** (without using high-level machine learning libraries like `scikit-learn`) to predict maximum daily AQI values and identify key pollution drivers.

---

## 📊 Dataset & Features
* **Source:** [Kaggle - Air Quality Index in Jakarta (2010–2021)](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021) (`ispu_dki_all.csv`)
* **Cleaned Dataset Size:** 4,875 records

### Features & Target
* **Target Variable ($y$):**
  * `max` — Maximum Air Quality Index (Daily AQI)
* **Independent Variables ($X$):**
  * `pm10` — Particulate Matter ($\le 10 \mu m$)
  * `so2` — Sulfur Dioxide
  * `co` — Carbon Monoxide
  * `o3` — Ground-level Ozone (*Strongest correlation with AQI: $r = 0.84$*)
  * `no2` — Nitrogen Dioxide

---

## 🛠️ Data Preprocessing & Validation
1. **Cleaning:** Stripped leading/trailing whitespaces and converted text values to numeric floating-point types.
2. **Date Parsing:** Converted string dates to standard `datetime` formats.
3. **Validation:** Removed `NaN` values and deduplicated records by date, verifying non-negative constraints across all pollutant columns.
4. **Feature Scaling:** Applied standard score normalization ($\mu = 0, \sigma = 1$) fitted exclusively on training set parameters to prevent data leakage.

---

## ⚙️ Model Implementation
* **Algorithm:** Multiple Linear Regression built from scratch using `NumPy`.
* **Optimization:** Gradient Descent.
* **Data Split:** 80% Training ($n = 3,900$) / 20% Testing ($n = 975$) with standard random seed shuffling (`seed = 42`).

### Model Equation Parameters
$$\text{Predicted AQI} = \beta_0 + \beta_1(\text{pm10}) + \beta_2(\text{so2}) + \beta_3(\text{co}) + \beta_4(\text{o3}) + \beta_5(\text{no2})$$

* **Intercept ($\beta_0$):** `97.83`
* **Feature Weights:**
  * **Ozone (`o3`):** `+37.75` *(Primary driver of daily AQI)*
  * **Particulate Matter (`pm10`):** `+9.21`
  * **Sulfur Dioxide (`so2`):** `+7.35`
  * **Nitrogen Dioxide (`no2`):** `+4.62`
  * **Carbon Monoxide (`co`):** `-1.47`

---

## 📈 Performance & Results

Evaluating model accuracy on unseen test data using Mean Squared Error (**MSE**) and Coefficient of Determination (**$R^2$**):

| Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **MSE** | 276.94 | 337.03 |
| **$R^2$ Score** | 0.8391 | 0.8178 |

> **Key Finding:** The test $R^2$ score of **0.8178** shows that **~81.8% of the variance** in Jakarta's daily AQI can be explained by the 5 tracked pollutant concentrations alone.

---

## 🎛️ Hyperparameter Tuning
We tested learning rates ($\alpha \in [0.001, 0.05]$) and iterations ($5,000 \to 10,000$):

| Learning Rate ($\alpha$) | Iterations | Test MSE | Test $R^2$ |
| :--- | :--- | :--- | :--- |
| 0.001 | 5,000 | 337.9717 | 0.8173 |
| **0.005 (Best)** | **5,000** | **337.0271** | **0.8178** |
| 0.01 | 5,000 | 337.0274 | 0.8178 |
| 0.01 | 10,000 | 337.0274 | 0.8178 |
| 0.05 | 5,000 | 337.0274 | 0.8178 |

*The minimal shift in MSE across hyperparameter tests confirms that the original Gradient Descent optimization had fully converged.*

---

## 💡 Insights & Public Health Impact
* **Primary Pollutant Driver:** Ground-level Ozone (`o3`) exhibits the highest correlation and model weight, indicating photochemical smog is a critical component of hazardous AQI days in Jakarta.
* **Actionable Interventions:** While the model cannot directly pinpoint point-source emissions, environmental agencies can use peak `o3` and `pm10` trends to issue targeted public safety warnings and investigate underlying contributors (e.g., traffic congestion, industrial output).