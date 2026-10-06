# EV Charging Forecasting Dashboard

This project provides an interactive Streamlit dashboard to explore historical electric vehicle (EV) adoption and forecast future county-level trends.

## Key Features

- State and county selection for localized analysis
- Adjustable forecast horizon (1 to 5 years)
- Single-county cumulative EV trend visualization (historical + forecast)
- Multi-county comparison (up to 3 counties)
- Monthly forecast table with CSV download
- Yearly forecast summary with projected totals
- Quick insight metrics for current EV share and projected growth

## Project Files

- `app.py` - Streamlit application
- `forecasting_ev_model.pkl` - Trained forecasting model
- `preprocessed_ev_data.csv` - Preprocessed monthly EV dataset used by the app
- `EV-demind-forecast01.ipynb` - Notebook used during model/data development

## Run Locally

1. Install dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib scikit-learn joblib
   ```
2. Start the app from the repository root:
   ```bash
   streamlit run app.py
   ```

## Notes

- The app forecasts EV totals using engineered lag and growth-based features.
- Forecast outputs are model-driven estimates and should be interpreted as directional planning insights.
