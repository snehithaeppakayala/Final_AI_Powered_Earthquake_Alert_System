## Project Title : AI-Powered Earthquake Alert System

##  Project Overview
This project predicts **earthquake alert levels** (Green, Yellow, Orange, Red) using **Machine Learning** techniques.  
We use features such as **latitude, longitude, depth, and magnitude** to classify earthquake severity levels.  

The project includes:
- Data preprocessing  
- Feature scaling  
- Handling imbalanced dataset using **SMOTE**  
- Training a **Random Forest Classifier**  
- Model evaluation with accuracy, classification report, and confusion matrix  
- Saving trained model using **Joblib**  


##  Features
- Predicts earthquake alert level (Green, Yellow, Orange, Red).  
- Balances dataset using SMOTE.  
- Visualizes confusion matrix.  
- Saves trained model, scaler, and label encoder for later deployment.  


##  Project Structure
AI-Powered Earthquake Alert System/
|-- templates/
|     |____index.html   # Web Interface
│-- earthquakes.csv # Dataset (from Kaggle)
│-- Alert.py # Main training script
|-- app.py   #Flask backend
│-- earthquake_model.pkl # Saved Random Forest model
│-- scaler.pkl # Saved StandardScaler
│-- label_encoder.pkl # Saved LabelEncoder
│-- README.md # Project documentation


##  Installation
Install required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib

Usage

Place your dataset earthquakes.csv in the project folder.
Run the training script:

python Alert.py

The model, scaler, and label encoder will be saved as .pkl files.
You will also get evaluation metrics (accuracy, classification report, confusion matrix).

Example Output
Accuracy: 0.98

Classification Report:
              precision    recall  f1-score   support
       green       0.99      0.96      0.98
      orange       0.98      0.99      0.99
         red       0.98      1.00      0.99
      yellow      0.96      0.97      0.97

Visualizations 

Confusion Martix Visualization

![alt text](<Screenshot 2025-08-28 184839.png>)

Alert Distribution in Bar Chart

![alt text](<Screenshot 2025-09-06 092951.png>)

Alert Distribution in Pie Chart

![alt text](<Screenshot 2025-09-06 093016.png>)

Magnitude vs Depth by Alert Level

![alt text](<Screenshot 2025-09-06 093042.png>)

ROC Curve (Multi-class)

![alt text](<Screenshot 2025-09-06 093105.png>)

Real-Time Prediction

The script allows user input for live earthquake alert prediction:

--- Real-Time Earthquake Alert Prediction ---
Enter Latitude: 15.2
Enter Longitude: 78.4
Enter Depth (km): 25
Enter Magnitude: 7.5

Predicted Earthquake Alert Level: orange
Alert Description: High alert, potential damage, take caution.

Red Alert Override: Any earthquake with Magnitude ≥ 8.0 & Depth ≤ 30 km is automatically classified as red.

Run Flask App
python app.py
Then open browser at  http://127.0.0.1:5000/

Output:

![alt text](<Screenshot 2025-09-09 235042.png>)

⚠️ Alert Levels & Meaning

🟢 Green → Minor earthquake, low risk
🟡 Yellow → Moderate earthquake, some risk of damage
🟠 Orange → High alert, potential damage, take caution
🔴 Red → Severe earthquake, emergency required

Future Enhancements:

 Integration with real-time seismic data APIs
 Mobile/Email alert notifications
 Deployment on Heroku / AWS / GCP
 