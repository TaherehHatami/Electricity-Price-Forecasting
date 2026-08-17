# Day-Ahead Electricity Price Forecasting
A probabilistic forecasting project predicting day-ahead electricity prices for the Germany-Luxembourg market, built as a technical take-home task for a data science interview.

📄 Full code: [view notebook](EPF-TaherehHatami.ipynb) | 📊 Slides: [view PDF](EPF-TaherehHatami.pdf)

## 📌 Project Overview
This project builds a day-ahead electricity price forecasting model for the German-Luxembourg electricity market, with a focus on producing not just point forecasts but **probabilistic** predictions; quantifying uncertainty, not just a single predicted price. The goal was to model price behavior realistically, including negative prices and extreme volatility events, which are common in real electricity markets driven by renewable oversupply.

## 📂 Dataset
- **Source:** Hourly electricity market data, Germany-Luxembourg bidding zone 
- **Size:** 92,519 hourly records (Jan 2015 – Jul 2025), 46 features after engineering
- **Note:** Prices range from −500 to +936 €/MWh, including 2,363 negative-price hours — a result of renewable oversupply events. Missing values in commodity columns were forward-filled; columns with heavy missingness (e.g. actuals, day-ahead temperature at 59% missing) were dropped.

## 🛠️ Tools Used
- Python
- pandas — data cleaning and feature engineering
- scikit-learn — LASSO regression, quantile regression, TimeSeriesSplit cross-validation
- matplotlib — visualization

## 🔍 Process
1. **Data Overview** — explored 92,519 hourly records; identified negative prices and heavy right-skew, requiring a transformed target variable.
2. **Feature Engineering** — built price lag features (1h, 24h, 48h, 168h), a residual load variable (Load − Solar − Wind), marginal cost indicators (gas, coal, oil, EUA, EUR/USD) and calendar dummies (hour of day, weekday).
3. **Target Transformation** — applied an arcsinh transformation to the price target to handle negative values while reducing skewness.
4. **Modeling — LEAR (LASSO-Estimated AutoRegressive Model)** — used LASSO regression for automatic feature selection, tuning the regularization strength (λ) via TimeSeriesSplit cross-validation. Trained on 2015–2022 data (69,865 hours), validated on 2023.
5. **Probabilistic Forecasting** — extended the model using quantile regression to produce prediction intervals (q10–q90), not just single-point forecasts, tuning quantile-specific λ via pinball loss minimization.
6. **Error Analysis** — examined prediction errors by hour of day and by month to identify when the model struggles most.

## 💡 Key Findings
- **Lag₁ (price one hour ago) dominates the model** — coefficient of 1.12, and removing it increases error by 3.5×, confirming strong short-term price persistence.
- LASSO selected 40 of 46 engineered features, automatically removing 6 as irrelevant.
- **Residual load** (load minus renewable generation) is a strong signal, capturing real-time supply-demand imbalance.
- The model struggled most with **extreme events** — e.g. a −500 €/MWh price in July 2023 driven by extreme solar oversupply - showing the limits of a linear model during rare, extreme conditions.
- **Prediction errors peak in the evening (17h–20h)** and in **January, February, and September** — likely tied to higher demand volatility and heating/cooling season transitions.
- **Validation results:** MAE 10.58 €/MWh, RMSE 16.21 €/MWh, 74.9% coverage (q10–q90, expected 80%)
- **Test results (2024):** MAE 9.97 €/MWh, RMSE 18.46 €/MWh, 56.4% coverage — the drop in coverage shows the model underestimates uncertainty during more volatile periods, meaning prediction intervals were too narrow.

## 🔮 Future Work
- Explore non-linear models (e.g. gradient boosting) to better capture extreme price events
- Combine multiple models using ensemble approaches
- Improve uncertainty estimation, especially prediction interval width during high-volatility periods
- Better handling of extreme/negative price spikes driven by renewable oversupply
