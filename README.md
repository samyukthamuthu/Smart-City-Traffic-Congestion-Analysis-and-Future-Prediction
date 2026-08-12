# Smart City Traffic Congestion Prediction

A Streamlit application for analyzing simulated Chennai-based traffic data and predicting future congestion levels using a Random Forest classifier.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_combined.py
```

## Files

- `app_combined.py` - Streamlit application
- `traffic_data_realistic_2024_2026_3456.csv` - traffic dataset
- `future_congestion_model.pkl` - trained Random Forest model
- `encoders.pkl` - LabelEncoder objects
- `requirements.txt` - deployment dependencies

## Note

The dataset is simulated Chennai-based data and should not be presented as official traffic-sensor data.
