import streamlit as st
import pandas as pd
import pickle
from datetime import date, time

st.set_page_config(page_title="Smart City Traffic Congestion Prediction",
                   page_icon="🚦", layout="wide")

DATA_FILE="traffic_data_realistic_2024_2026_3456.csv"
MODEL_FILE="future_congestion_model.pkl"
ENCODER_FILE="encoders.pkl"

@st.cache_data
def load_data():
    data=pd.read_csv(DATA_FILE)
    data["Date"]=pd.to_datetime(data["Date"],errors="coerce")
    return data

@st.cache_resource
def load_artifacts():
    with open(MODEL_FILE,"rb") as f:
        model=pickle.load(f)
    with open(ENCODER_FILE,"rb") as f:
        encoders=pickle.load(f)
    return model,encoders

try:
    df=load_data()
    model,encoders=load_artifacts()
except Exception as e:
    st.error("Unable to load the dataset or model files.")
    st.exception(e)
    st.stop()

st.sidebar.title("🚦 Smart City Traffic")
page=st.sidebar.radio("Navigation",["Dashboard","Future Prediction"])

if page=="Dashboard":
    st.title("📊 Smart City Traffic Congestion Dashboard")
    st.caption("Simulated Chennai-based traffic dataset")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Records",f"{len(df):,}")
    c2.metric("Average Speed",f"{df['Speed'].mean():.1f} km/h")
    c3.metric("Average Traffic",f"{df['Traffic Volume'].mean():.0f}")
    c4.metric("Accidents",int((df["Accident"]=="Yes").sum()))

    st.subheader("Traffic Volume Trend")
    trend=df.groupby("Date",as_index=False)["Traffic Volume"].mean().set_index("Date")
    st.line_chart(trend)

    col1,col2=st.columns(2)
    with col1:
        st.subheader("Average Traffic by Vehicle Type")
        vehicle=df.groupby("Vehicle Type")["Traffic Volume"].mean().sort_values(ascending=False)
        st.bar_chart(vehicle)
    with col2:
        st.subheader("Congestion Distribution")
        st.bar_chart(df["Congestion Level"].value_counts())

    st.subheader("Average Traffic by Weather")
    weather=df.groupby("Weather")["Traffic Volume"].mean().sort_values(ascending=False)
    st.bar_chart(weather)

    st.subheader("Data Preview")
    st.dataframe(df.head(20),use_container_width=True)

else:
    st.title("🔮 Future Congestion Prediction")
    st.write("Enter expected traffic conditions to predict congestion.")

    d=st.date_input("Future Date",value=date.today())
    t=st.time_input("Time",value=time(8,0))
    location=st.selectbox("Location",list(encoders["Location"].classes_))
    traffic=st.number_input("Traffic Volume",min_value=0,max_value=5000,value=500,step=10)
    vehicle=st.selectbox("Vehicle Type",list(encoders["Vehicle Type"].classes_))
    weather=st.selectbox("Weather",list(encoders["Weather"].classes_))
    temp=st.number_input("Temperature",min_value=-10.0,max_value=60.0,value=30.0,step=0.5)
    holiday=st.selectbox("Holiday",list(encoders["Holiday"].classes_))
    speed=st.number_input("Speed",min_value=0.0,max_value=150.0,value=40.0,step=1.0)
    accident=st.selectbox("Accident",list(encoders["Accident"].classes_))

    if st.button("🚦 Predict Congestion",type="primary"):
        try:
            X=pd.DataFrame({
                "Location":[encoders["Location"].transform([location])[0]],
                "Traffic Volume":[traffic],
                "Vehicle Type":[encoders["Vehicle Type"].transform([vehicle])[0]],
                "Weather":[encoders["Weather"].transform([weather])[0]],
                "Temperature":[temp],
                "Holiday":[encoders["Holiday"].transform([holiday])[0]],
                "Speed":[speed],
                "Accident":[encoders["Accident"].transform([accident])[0]],
                "Year":[d.year],"Month":[d.month],"Day":[d.day],
                "DayOfWeek":[d.weekday()],
                "WeekOfYear":[d.isocalendar().week],
                "Hour":[t.hour]
            })
            pred=model.predict(X)
            label=encoders["Congestion Level"].inverse_transform(pred)[0]
            st.success(f"Predicted Congestion: {label}")
            if hasattr(model,"predict_proba"):
                conf=model.predict_proba(X).max()*100
                st.metric("Prediction Confidence",f"{conf:.2f}%")
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
