import base64

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="EV Forecast", layout="wide")


# === Set blurred background image ===
def set_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
            }}
            .main-title {{
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #ffffff;
                margin-top: 20px;
                text-shadow: 2px 2px 4px #000;
            }}
            .subtitle {{
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                color: #eeeeee;
                margin-bottom: 25px;
                text-shadow: 1px 1px 3px #000;
            }}
            .instruction {{
                font-size: 20px;
                color: #ffffff;
                padding: 10px;
                background-color: rgba(0, 0, 0, 0.4);
                border-radius: 8px;
                margin-top: 10px;
                text-align: center;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(6px);
                border-radius: 15px;
                padding: 1rem 2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    data = pd.read_csv("preprocessed_ev_data.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def build_features(months_since_start, county_code, historical_ev, cumulative_ev):
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) >= 2 else 0

    return {
        "months_since_start": months_since_start,
        "county_encoded": county_code,
        "ev_total_lag1": lag1,
        "ev_total_lag2": lag2,
        "ev_total_lag3": lag3,
        "ev_total_roll_mean_3": roll_mean,
        "ev_total_pct_change_1": pct_change_1,
        "ev_total_pct_change_3": pct_change_3,
        "ev_growth_slope": ev_growth_slope,
    }


def generate_forecast(model, county_df, horizon_months):
    county_df = county_df.sort_values("Date")
    county_code = county_df["county_encoded"].iloc[0]
    historical_ev = list(county_df["Electric Vehicle (EV) Total"].astype(float).values)

    if len(historical_ev) < 3:
        seed = historical_ev[-1] if historical_ev else 0
        while len(historical_ev) < 3:
            historical_ev.insert(0, seed)

    cumulative_ev = list(np.cumsum(historical_ev[-6:]))
    months_since_start = int(county_df["months_since_start"].max())
    latest_date = county_df["Date"].max()

    recent_history = historical_ev[-6:]
    future_rows = []

    for i in range(1, horizon_months + 1):
        months_since_start += 1
        features = build_features(months_since_start, county_code, recent_history, cumulative_ev)
        pred = max(0, model.predict(pd.DataFrame([features]))[0])

        forecast_date = latest_date + pd.DateOffset(months=i)
        future_rows.append({"Date": forecast_date, "Predicted EV Total": int(round(pred))})

        recent_history.append(float(pred))
        if len(recent_history) > 6:
            recent_history.pop(0)

        cumulative_ev.append(cumulative_ev[-1] + float(pred))
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    forecast_df = pd.DataFrame(future_rows)
    historical_cum = county_df[["Date", "Electric Vehicle (EV) Total"]].copy()
    historical_cum["Cumulative EV"] = historical_cum["Electric Vehicle (EV) Total"].cumsum()

    forecast_df["Cumulative EV"] = forecast_df["Predicted EV Total"].cumsum() + historical_cum["Cumulative EV"].iloc[-1]
    return historical_cum, forecast_df


# === App bootstrap ===
set_bg_from_local("bg-ev.jpg")
model = joblib.load("forecasting_ev_model.pkl")
df = load_data()

st.markdown("<div class='main-title'>🔮 EV Adoption Forecaster for County-Level Trends</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Explore historical EV growth and model-driven projections by state and county.</div>", unsafe_allow_html=True)
st.markdown("<div class='instruction'>Pick a state, choose counties, and forecast EV adoption for the next 1 to 5 years.</div>", unsafe_allow_html=True)

# === Sidebar controls ===
st.sidebar.header("Forecast Controls")
state_list = sorted(df["State"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Select State", state_list)
forecast_years = st.sidebar.slider("Forecast horizon (years)", min_value=1, max_value=5, value=3)
forecast_horizon = forecast_years * 12

state_df = df[df["State"] == selected_state]
county_list = sorted(state_df["County"].dropna().unique().tolist())

if not county_list:
    st.warning(f"No county data found for state '{selected_state}'.")
    st.stop()

# === Single county deep dive ===
st.header("County Forecast")
selected_county = st.selectbox("Select a County", county_list)
county_df = state_df[state_df["County"] == selected_county]

historical_cum, forecast_df = generate_forecast(model, county_df, forecast_horizon)
historical_cum["Source"] = "Historical"
forecast_df["Source"] = "Forecast"

combined = pd.concat(
    [
        historical_cum[["Date", "Cumulative EV", "Source"]],
        forecast_df[["Date", "Cumulative EV", "Source"]],
    ],
    ignore_index=True,
)

st.subheader(f"📊 Cumulative EV Forecast for {selected_county}, {selected_state}")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby("Source"):
    ax.plot(data["Date"], data["Cumulative EV"], label=label, marker="o")

ax.set_title(f"Cumulative EV Trend - {selected_county} ({forecast_years}-Year Forecast)", fontsize=14, color="white")
ax.set_xlabel("Date", color="white")
ax.set_ylabel("Cumulative EV Count", color="white")
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor("#1c1c1c")
ax.tick_params(colors="white")
ax.legend()
st.pyplot(fig)

latest_row = county_df.sort_values("Date").iloc[-1]
historical_total = float(historical_cum["Cumulative EV"].iloc[-1])
forecasted_total = float(forecast_df["Cumulative EV"].iloc[-1])
forecast_growth_pct = ((forecasted_total - historical_total) / historical_total * 100) if historical_total > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Latest Monthly EV Total", f"{int(latest_row['Electric Vehicle (EV) Total']):,}")
col2.metric("Current EV Share", f"{latest_row['Percent Electric Vehicles']:.2f}%")
col3.metric(
    f"{forecast_years}-Year Growth",
    f"{forecast_growth_pct:.2f}%",
    delta=f"{int(forecasted_total - historical_total):,} cumulative EV",
)

if historical_total > 0:
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(
        f"Based on the selected horizon, EV adoption in **{selected_county}** is expected to show a **{trend} of {forecast_growth_pct:.2f}%**."
    )

st.markdown("### Monthly Forecast Table")
display_forecast = forecast_df[["Date", "Predicted EV Total", "Cumulative EV"]].copy()
display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
display_forecast["Cumulative EV"] = display_forecast["Cumulative EV"].round(0).astype(int)
st.dataframe(display_forecast, use_container_width=True)

st.download_button(
    label="Download Forecast CSV",
    data=display_forecast.to_csv(index=False),
    file_name=f"ev_forecast_{selected_state}_{selected_county}_{forecast_years}y.csv".replace(" ", "_"),
    mime="text/csv",
)

st.markdown("### Yearly Forecast Summary")
yearly_summary = forecast_df.copy()
yearly_summary["Year"] = yearly_summary["Date"].dt.year
yearly_summary = (
    yearly_summary.groupby("Year", as_index=False)
    .agg(
        Predicted_EV_Total=("Predicted EV Total", "sum"),
        End_of_Year_Cumulative_EV=("Cumulative EV", "max"),
    )
    .astype({"Predicted_EV_Total": int, "End_of_Year_Cumulative_EV": int})
)
st.dataframe(yearly_summary, use_container_width=True)

# === Multi-county comparison ===
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")

multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []
    growth_summaries = []

    for cty in multi_counties:
        cty_df = state_df[state_df["County"] == cty]
        hist_cum, fc_df = generate_forecast(model, cty_df, forecast_horizon)

        combined_cty = pd.concat(
            [hist_cum[["Date", "Cumulative EV"]], fc_df[["Date", "Cumulative EV"]]],
            ignore_index=True,
        )
        combined_cty["County"] = cty
        comparison_data.append(combined_cty)

        hist_total = float(hist_cum["Cumulative EV"].iloc[-1])
        fc_total = float(fc_df["Cumulative EV"].iloc[-1])
        if hist_total > 0:
            growth_pct = ((fc_total - hist_total) / hist_total) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A")

    comp_df = pd.concat(comparison_data, ignore_index=True)

    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby("County"):
        ax.plot(group["Date"], group["Cumulative EV"], marker="o", label=cty)
    ax.set_title(f"EV Adoption Trends: Historical + {forecast_years}-Year Forecast", fontsize=16, color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Cumulative EV Count", color="white")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor("#1c1c1c")
    ax.tick_params(colors="white")
    ax.legend(title="County")
    st.pyplot(fig)

    summary = " | ".join(growth_summaries)
    st.success(f"Forecasted EV adoption growth over next {forecast_years} years — {summary}")

st.markdown("Prepared for the **AICTE Internship Cycle 2 by S4F**")
