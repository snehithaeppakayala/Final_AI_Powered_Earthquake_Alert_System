from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Load trained model, scaler, label encoder
clf = joblib.load("earthquake_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Optional: alert descriptions
alert_desc = {
    'green': 'Minor earthquake, low risk.',
    'yellow': 'Moderate earthquake, some risk of damage.',
    'orange': 'High alert, potential damage, take caution.',
    'red': 'Severe earthquake, major damage possible, emergency required.'
}

# Load dataset for charts
df = pd.read_csv('earthquakes.csv')
df = df.dropna(subset=['latitude','longitude','depth','mag','Alert'])

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    description = None
    
    if request.method == "POST":
        try:
            lat = float(request.form["latitude"])
            lon = float(request.form["longitude"])
            depth = float(request.form["depth"])
            mag = float(request.form["magnitude"])
            
            user_input = pd.DataFrame([[lat, lon, depth, mag]], columns=['latitude','longitude','depth','mag'])
            user_input_scaled = scaler.transform(user_input)
            
            # Red Alert Override
            if mag >= 8.0 and depth <= 30:
                alert_label = ['red']
            else:
                predicted_alert = clf.predict(user_input_scaled)
                alert_label = le.inverse_transform(predicted_alert)
            
            prediction = alert_label[0]
            description = alert_desc.get(prediction, "Unknown")
        except:
            prediction = "Error"
            description = "Invalid input. Please enter numeric values."
    
    # Data for charts
    alert_counts = df['Alert'].value_counts().to_dict()
    
    return render_template("index.html", prediction=prediction, description=description, alert_counts=alert_counts)

if __name__ == "__main__":
    app.run(debug=True)