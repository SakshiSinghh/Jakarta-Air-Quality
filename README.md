# Jakarta Air Quality Index (AQI) Analysis & Prediction

A machine learning project developed for **01.020 Design Thinking Project III (DTP III)** to analyze atmospheric pollutants in Jakarta and predict maximum daily Air Quality Index (AQI) levels.

---

## 📌 Project Overview

### 🌆 Background
Air pollution in urban centers like Jakarta poses severe public health risks. By modelling the relationship between major atmospheric pollutants ($\text{PM}_{10}$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, $\text{NO}_2$) and daily peak Air Quality Index (AQI), health authorities can better understand the primary drivers of poor air quality, identify key emission sources, and implement targeted interventions to protect public health.

---

### 👤 User Persona

* **Name:** Aisyah
* **Occupation:** Environmental Monitoring Officer
* **Goals:**
  * Evaluate daily Jakarta air quality conditions in real-time.
  * Identify which specific pollutant sources are driving poor air quality.
  * Provide data-driven recommendations to help policymakers target various emission sources for public health interventions.
* **Pain Points:**
  * Air pollution sources are complex, making it difficult to manually determine which pollutants are driving up the AQI.
  * Policymakers need clear, empirical data on pollutant contributions rather than guesswork to design effective, targeted air quality policies.

---

### 🎯 Problem Statement
> **"How do specific pollutant concentrations contribute to Jakarta’s overall Air Quality Index (AQI), and how can this data be used to target specific emission sources for public health interventions?"**

Using daily AQI data from 2010 to 2021, we implemented a **Multiple Linear Regression model trained via Gradient Descent using NumPy** to quantify pollutant impacts and predict daily maximum AQI levels.

---

## 📊 Dataset & Features
* **Source:** [Kaggle - Air Quality Index in Jakarta (2010–2021)](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021) (`ispu_dki_all.csv`)
* **Cleaned Dataset Size:** 4,875 daily records

### Variables
* **Target Variable ($y$):**
  * `max` — Maximum Air Quality Index (Daily AQI)
* **Predictor Variables ($X$):**
  * `pm10` — Particulate Matter ($\le 10 \mu m$)
  * `so2` — Sulfur Dioxide
  * `co` — Carbon Monoxide
  * `o3` — Ground-level Ozone (*Strongest correlation with target: $r = 0.845$*)
  * `no2` — Nitrogen Dioxide

---

## 🛠️ Data Preprocessing & Validation
1. **Cleaning:** Stripped whitespaces and converted raw strings into numeric float types.
2. **Date Parsing:** Converted date strings into standard `datetime` objects.
3. **Validation:** Removed `NaN` values, deduplicated records by date, and verified non-negative constraints across pollutant readings.
4. **Feature Normalization:** Standardized predictors using standard score scaling ($\mu = 0, \sigma = 1$). To prevent data leakage, scaling parameters were computed **only on the training split**.

---

## ⚙️ Model & Implementation
* **Model Type:** Multiple Linear Regression
* **Optimization:** Gradient Descent (implemented via matrix operations in `NumPy`)
* **Data Split:** 80% Training ($n = 3,900$) / 20% Testing ($n = 975$) with random seed shuffling (`seed = 42`)

### Learned Model Parameters
$$\text{Predicted AQI} = \beta_0 + \beta_1(\text{pm10}) + \beta_2(\text{so2}) + \beta_3(\text{co}) + \beta_4(\text{o3}) + \beta_5(\text{no2})$$

* **Intercept ($\beta_0$):** `97.83`
* **Standardized Weights:**
  * **Ozone (`o3`):** `+37.75` *(Primary contributor to daily peak AQI)*
  * **Particulate Matter (`pm10`):** `+9.21`
  * **Sulfur Dioxide (`so2`):** `+7.35`
  * **Nitrogen Dioxide (`no2`):** `+4.62`
  * **Carbon Monoxide (`co`):** `-1.47`

---

## 📈 Model Performance & Evaluation

We evaluated model accuracy on unseen test data using Mean Squared Error (**MSE**) and Coefficient of Determination (**$R^2$**):

| Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **MSE** | 276.94 | 337.03 |
| **$R^2$ Score** | 0.8391 | 0.8178 |

> **Key Finding:** The test $R^2$ score of **0.8178** shows that **~81.8% of the variance** in Jakarta's daily maximum AQI can be explained by these 5 tracked pollutant concentrations.

---

## 🎛️ Hyperparameter Iterations
We systematically tested learning rates ($\alpha \in [0.001, 0.05]$) and gradient descent iterations ($5,000 \to 10,000$) to test convergence:

| Learning Rate ($\alpha$) | Iterations | Test MSE | Test $R^2$ |
| :--- | :--- | :--- | :--- |
| 0.001 | 5,000 | 337.9717 | 0.8173 |
| **0.005 (Selected)** | **5,000** | **337.0271** | **0.8178** |
| 0.01 | 5,000 | 337.0274 | 0.8178 |
| 0.01 | 10,000 | 337.0274 | 0.8178 |
| 0.05 | 5,000 | 337.0274 | 0.8178 |

*The plateau in test MSE indicates that Gradient Descent achieved full optimization convergence by 5,000 iterations.*

---

## 💡 Insights & Public Health Impact
* **Primary Contributor:** Ground-level Ozone (`o3`) shows the strongest positive correlation ($r = 0.845$) and highest model weight ($\beta = +37.75$), indicating that photochemical reactions are the driving force behind Jakarta's highest AQI days.
* **Targeted Interventions:** Public health authorities can prioritize controlling precursor emissions (like vehicular exhausts and industrial volatile organic compounds) that react to form ozone, alongside sending public safety alerts on high `o3` and `pm10` days.
* **Limitations & Future Work:** The remaining ~18.2% of unexplained variance suggests that weather factors (such as sunlight, humidity, and wind speed) also influence daily AQI. Future models could incorporate meteorological data.