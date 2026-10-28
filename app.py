"""Olympic swimming history and cautious 2028 forecasting."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="Olympic Swimming Predictor", page_icon="🏊", layout="wide")
st.title("🏊 Olympic Swimming: History & 2028 Predictions")
st.caption("Fastest recorded results in the supplied dataset; forecasts are exploratory, not guarantees.")

DATA_PATH = Path(__file__).with_name("Olympic_Swimming_1912-2024.csv")


def to_seconds(value):
    """Convert SS.ss, MM:SS.ss, or HH:MM:SS.ss to seconds."""
    if pd.isna(value):
        return np.nan
    try:
        parts = [float(p.strip()) for p in str(value).strip().split(":")]
        if len(parts) > 3 or not parts or any(not np.isfinite(p) or p < 0 for p in parts):
            return np.nan
        seconds = sum(part * 60 ** i for i, part in enumerate(reversed(parts)))
        return seconds if seconds > 0 else np.nan
    except ValueError:
        return np.nan


def format_time(seconds):
    if pd.isna(seconds):
        return "N/A"
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes}:{remainder:05.2f}" if minutes else f"{remainder:.2f} s"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Seconds"] = df["Results"].apply(to_seconds)
    df["Stroke"] = df["Stroke"].replace({"Individual medley": "Individual Medley", "Medley": "Medley Relay"})
    df["Distance (in meters)"] = df["Distance (in meters)"].replace({"4x100": "4x100m", "4x200": "4x200m"})
    df["Team"] = df["Team"].astype(str).str.strip()
    df = df.dropna(subset=["Seconds", "Year"])
    df["Year"] = df["Year"].astype(int)
    return df


def model_factory():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=2, random_state=42),
    }


def fit_predict(model, train_years, train_times, prediction_years):
    # Years are shifted/scaled for numerical stability; models receive ONLY the year.
    X = ((np.asarray(train_years, dtype=float) - 2000) / 4).reshape(-1, 1)
    X_future = ((np.asarray(prediction_years, dtype=float) - 2000) / 4).reshape(-1, 1)
    model.fit(X, np.asarray(train_times, dtype=float))
    return model.predict(X_future)


def evaluate_models(history, forecast_year=2028):
    history = history.sort_values("Year")
    years = history["Year"].to_numpy()
    times = history["Seconds"].to_numpy()
    if len(years) < 5:
        return None, None, None

    # Expanding-window backtesting: at each step, train only on earlier Olympics.
    scores = []
    for name in model_factory():
        actual, predicted = [], []
        for index in range(3, len(years)):
            model = model_factory()[name]
            guess = float(fit_predict(model, years[:index], times[:index], [years[index]])[0])
            if np.isfinite(guess) and guess > 0:
                actual.append(float(times[index]))
                predicted.append(guess)
        if not actual:
            continue
        errors = np.abs((np.asarray(actual) - np.asarray(predicted)) / np.asarray(actual)) * 100
        scores.append({"Model": name, "MAPE (%)": float(np.mean(errors)), "Backtest predictions": len(errors)})

    if not scores:
        return None, None, None
    table = pd.DataFrame(scores).sort_values("MAPE (%)").reset_index(drop=True)
    winner = table.iloc[0]["Model"]
    forecast = float(fit_predict(model_factory()[winner], years, times, [forecast_year])[0])
    if not np.isfinite(forecast) or forecast <= 0:
        return table, winner, None
    return table, winner, forecast


try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Cannot find {DATA_PATH.name}. Put the CSV beside app.py.")
    st.stop()

with st.sidebar:
    st.header("Select an Olympic event")
    gender = st.selectbox("Gender", ["Men", "Women"])
    gender_df = df[(df["Gender"] == gender) & (df["Relay?"] == 0)].copy()
    distances = sorted(gender_df["Distance (in meters)"].unique(), key=lambda s: int(s.rstrip("m")))
    distance = st.selectbox("Distance", distances)
    distance_df = gender_df[gender_df["Distance (in meters)"] == distance]
    stroke = st.selectbox("Stroke", sorted(distance_df["Stroke"].unique()))
    event_df = distance_df[distance_df["Stroke"] == stroke]
    countries = sorted(event_df["Team"].unique())
    default = [c for c in ["USA", "AUS", "GBR"] if c in countries]
    selected = st.multiselect("Countries (choose one or more)", countries, default=default)
    st.info("Individual events only. Relay events are excluded in this first version.")

if not selected:
    st.warning("Select at least one country from the sidebar.")
    st.stop()

subset = event_df[event_df["Team"].isin(selected)]
history = (subset.groupby(["Team", "Year"], as_index=False)["Seconds"].min()
           .sort_values(["Team", "Year"]))

st.subheader(f"{gender} • {distance} {stroke}")
st.write("Each point is the fastest valid recorded result for that country in that Olympic year. Lower times are better.")
fig = go.Figure()
summary = []
score_tables = []
for country in selected:
    h = history[history["Team"] == country].sort_values("Year")
    if h.empty:
        continue
    scores, winner, forecast = evaluate_models(h)
    fig.add_trace(go.Scatter(x=h["Year"], y=h["Seconds"], mode="lines+markers",
                             name=f"{country} actual", connectgaps=False,
                             hovertemplate="Year %{x}<br>Time %{y:.2f} seconds<extra></extra>"))
    if forecast is not None:
        fig.add_trace(go.Scatter(x=[int(h["Year"].max()), 2028],
                                 y=[float(h.iloc[-1]["Seconds"]), forecast],
                                 mode="lines+markers", line=dict(dash="dash"),
                                 name=f"{country} predicted 2028"))
        score_table = scores.copy()
        score_table.insert(0, "Country", country)
        score_tables.append(score_table)
    summary.append({"Country": country, "Recorded Olympic years": len(h),
                    "Latest recorded year": int(h["Year"].max()),
                    "Latest recorded time": format_time(float(h.iloc[-1]["Seconds"])),
                    "Best model": winner or "Insufficient data",
                    "Backtest MAPE (%)": round(float(scores.iloc[0]["MAPE (%)"]), 2) if scores is not None and forecast is not None else np.nan,
                    "Predicted 2028 time": format_time(forecast) if forecast is not None else "Unavailable"})

fig.update_layout(xaxis_title="Olympic year", yaxis_title="Time (seconds) — lower is better",
                  hovermode="x unified", legend_title="Country / series", height=540)
st.plotly_chart(fig, use_container_width=True)
st.caption("Dashed segments are model forecasts. Straight connecting lines are visual guides, not results for missing Olympics.")

st.subheader("2028 predictions and model accuracy")
summary_df = pd.DataFrame(summary)
st.dataframe(summary_df, use_container_width=True, hide_index=True)
st.download_button("Download prediction summary (CSV)", summary_df.to_csv(index=False).encode("utf-8"),
                   file_name="swimming_predictions_2028.csv", mime="text/csv")

if score_tables:
    st.subheader("Compare all models")
    all_scores = pd.concat(score_tables, ignore_index=True)
    st.dataframe(all_scores.style.format({"MAPE (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.download_button("Download model errors (CSV)", all_scores.to_csv(index=False).encode("utf-8"),
                       file_name="swimming_model_errors.csv", mime="text/csv")

with st.expander("See the cleaned historical data"):
    st.dataframe(history, use_container_width=True, hide_index=True)

st.markdown("""
**How model selection works:** Each model is repeatedly trained on earlier observed Olympic years and
predicts a later observed year. Mean Absolute Percentage Error (MAPE) is averaged across those
out-of-sample predictions. The lowest-MAPE model is refitted on all observed years and forecasts 2028.
At least **5 recorded Olympic years** are required for a country/event combination.

**Limitations:** The CSV may omit participants, and it includes only available recorded results.
Forecasts extrapolate from time alone; they cannot account for new swimmers, rule changes, boycotts,
training innovations, or changing country participation. A low historical error does not guarantee
2028 accuracy. Tree models may predict a flat future value because they do not extrapolate trends.
""")
