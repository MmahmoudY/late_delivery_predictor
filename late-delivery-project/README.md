# Late-Delivery Predictor for an E-commerce Marketplace (Olist)

Predict, at purchase time, which orders will arrive after the estimated delivery date.

## Contents
- `late-delivery-predictor-for-an-e-commerce.ipynb` — full notebook (your original cells + feature engineering, EDA, modeling, seller scorecard, review-impact stretch goal, artifact export). Already executed, outputs included.
- `app/` — Streamlit app (order risk checker, seller scorecard, satisfaction impact, model info) + `artifacts/` (trained model & lookup tables).

## Run the app
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
## Headline results (out-of-time test = last 20% of orders)
- 96,447 delivered orders, 9 joined tables, 8.1% late
- Logistic Regression: ROC-AUC 0.72, PR-AUC 0.10 (naive baseline 0.053)
- Flag riskiest 20% of orders -> catches 38% of late orders (1.9x lift); app threshold flags 21% and catches ~40%
- Late orders get 1-2 stars 54% of the time vs 9% on time (5.9x)

