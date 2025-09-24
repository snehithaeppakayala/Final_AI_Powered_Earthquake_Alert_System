import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model, scaler, and label encoder
clf = joblib.load("earthquake_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Alert descriptions
alert_desc = {
    'green': '🟢 Minor earthquake, low risk.',
    'yellow': '🟡 Moderate earthquake, some risk of damage.',
    'orange': '🟠 High alert, potential damage, take caution.',
    'red': '🔴 Severe earthquake, major damage possible, emergency required.'
}

# Load dataset for visualization
df = pd.read_csv("earthquakes.csv")
df = df.dropna(subset=['latitude','longitude','depth','mag','Alert'])

# Streamlit Page Config
st.set_page_config(page_title="🌍 Earthquake Alert Prediction", page_icon="🌋", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1em;
            font-size: 1em;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌍 AI-Powered Earthquake Alert System")
st.write("An interactive ML-powered system to predict earthquake alert levels and visualize data trends.")

# Sidebar Inputs
st.sidebar.header("📝 Enter Earthquake Details")
lat = st.sidebar.number_input("🌐 Latitude", value=35.0, format="%.4f")
lon = st.sidebar.number_input("🌐 Longitude", value=-118.0, format="%.4f")
depth = st.sidebar.number_input("📏 Depth (km)", value=10.0, format="%.2f")
mag = st.sidebar.number_input("💥 Magnitude", value=5.5, format="%.2f")

if st.sidebar.button("🚨 Predict Alert Level"):
    user_input = pd.DataFrame([[lat, lon, depth, mag]], columns=['latitude','longitude','depth','mag'])
    user_input_scaled = scaler.transform(user_input)

    # Red Alert Override
    if mag >= 8.0 and depth <= 30:
        alert_label = ['red']
    else:
        predicted_alert = clf.predict(user_input_scaled)
        alert_label = le.inverse_transform(predicted_alert)

    st.markdown(f"""
        <div class="prediction-box" style="background-color:#87CE;">
            <p>Predicted Alert Level: <span style="color:#e74c3c;">{alert_label[0].capitalize()}</span></p>
            <p>{alert_desc.get(alert_label[0], "Unknown")}</p>
        </div>
    """, unsafe_allow_html=True)

