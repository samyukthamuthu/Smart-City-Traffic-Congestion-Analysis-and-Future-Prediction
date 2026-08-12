import streamlit as st
import pandas as pd
from datetime import date, time

st.set_page_config(
    page_title="Smart City Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide"
)

DATA_FILE = "traffic_data_realistic_2024_2026_3456.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Unable to load the dataset.")
    st.info("Keep traffic_data_realistic_2024_2026_3456.csv in the same GitHub folder as app_combined.py.")
    st.exception(e)
    st.stop()

# Clean values and create time features.
for col in ["Location", "Vehicle Type", "Weather", "Holiday", "Accident", "Congestion Level"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# This app intentionally uses only Streamlit + pandas.
# It does not require sklearn, joblib, plotly, or pickle.
#
# Prediction method:
# A small, transparent nearest-neighbour calculation is performed directly
# from the CSV. It compares the user's inputs with historical records and
# returns the majority congestion class among the closest records.

CAT_COLS = ["Location", "Vehicle Type", "Weather", "Holiday", "Accident"]
NUM_COLS = ["Traffic Volume", "Temperature", "Speed"]

# Pre-compute numeric ranges for distance normalization.
RANGES = {}
for c in NUM_COLS:
    lo = float(df[c].min())
    hi = float(df[c].max())
    RANGES[c] = max(hi - lo, 1.0)

CATEGORIES = {
    c: sorted(df[c].dropna().astype(str).unique().tolist())
    for c in CAT_COLS
}

def predict_congestion(location, traffic, vehicle, weather, temp,
                        holiday, speed, accident, pred_date, pred_time, k=31):
    """Return congestion label and confidence using historical nearest records."""
    q = {
        "Location": str(location),
        "Traffic Volume": float(traffic),
        "Vehicle Type": str(vehicle),
        "Weather": str(weather),
        "Temperature": float(temp),
        "Holiday": str(holiday),
        "Speed": float(speed),
        "Accident": str(accident),
        "Year": int(pred_date.year),
        "Month": int(pred_date.month),
        "Day": int(pred_date.day),
        "DayOfWeek": int(pred_date.weekday()),
        "WeekOfYear": int(pred_date.isocalendar().week),
        "Hour": int(pred_time.hour),
    }

    work = df.copy()

    # Numeric distance.
    dist = pd.Series(0.0, index=work.index)
    for c in NUM_COLS:
        dist += ((work[c].astype(float) - q[c]) / RANGES[c]) ** 2

    # Categorical mismatches add distance.
    for c in CAT_COLS:
        dist += (work[c].astype(str) != q[c]).astype(float) * 0.65

    # Calendar similarity; lower weight than traffic/speed.
    dist += ((work["Month"] - q["Month"]) / 12.0) ** 2 * 0.08
    dist += ((work["DayOfWeek"] - q["DayOfWeek"]) / 6.0) ** 2 * 0.08

    nearest = work.loc[dist.nsmallest(min(k, len(work))).index].copy()
    labels = nearest["Congestion Level"].astype(str)

    counts = labels.value_counts()
    label = counts.index[0]
    confidence = float(counts.iloc[0] / counts.sum() * 100)

    return label, confidence, nearest

st.sidebar.title("🚦 Smart City Traffic")
page = st.sidebar.radio("Navigation", ["Dashboard", "Future Prediction"])

if page == "Dashboard":
    st.title("📊 Smart City Traffic Congestion Dashboard")
    st.caption("Traffic congestion analysis and future prediction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Average Speed", f"{df['Speed'].mean():.1f} km/h")
    c3.metric("Average Traffic", f"{df['Traffic Volume'].mean():.0f}")

    accidents = (df["Accident"].str.lower() == "yes").sum()
    c4.metric("Accidents", f"{int(accidents):,}")

    st.subheader("Traffic Volume Trend")
    trend = df.groupby("Date")["Traffic Volume"].mean()
    st.line_chart(trend)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Traffic by Vehicle Type")
        vehicle = (
            df.groupby("Vehicle Type")["Traffic Volume"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(vehicle)

    with col2:
        st.subheader("Congestion Distribution")
        st.bar_chart(df["Congestion Level"].value_counts())

    st.subheader("Average Traffic by Weather")
    weather = (
        df.groupby("Weather")["Traffic Volume"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(weather)

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

else:
    st.title("🔮 Future Congestion Prediction")
    st.write("Enter expected traffic conditions to predict the congestion level.")

    d = st.date_input("Future Date", value=date.today())
    t = st.time_input("Time", value=time(8, 0))

    location = st.selectbox("Location", CATEGORIES["Location"])
    traffic = st.number_input(
        "Traffic Volume", min_value=0, max_value=5000, value=500, step=10
    )
    vehicle = st.selectbox("Vehicle Type", CATEGORIES["Vehicle Type"])
    weather = st.selectbox("Weather", CATEGORIES["Weather"])
    temp = st.number_input(
        "Temperature", min_value=-10.0, max_value=60.0, value=30.0, step=0.5
    )
    holiday = st.selectbox("Holiday", CATEGORIES["Holiday"])
    speed = st.number_input(
        "Speed", min_value=0.0, max_value=150.0, value=40.0, step=1.0
    )
    accident = st.selectbox("Accident", CATEGORIES["Accident"])

    if st.button("🚦 Predict Congestion", type="primary"):
        try:
            label, confidence, nearest = predict_congestion(
                location, traffic, vehicle, weather, temp,
                holiday, speed, accident, d, t
            )

            st.success(f"Predicted Congestion: {label}")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            with st.expander("See supporting historical records"):
                st.dataframe(
                    nearest[
                        ["Date", "Location", "Traffic Volume", "Vehicle Type",
                         "Weather", "Temperature", "Speed",
                         "Accident", "Holiday", "Congestion Level"]
                    ].head(10),
                    use_container_width=True
                )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
