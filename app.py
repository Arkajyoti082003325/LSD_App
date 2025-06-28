from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
import numpy as np

app = Flask(__name__)


model = joblib.load('models/landslide_model.pkl')
size_encoder = joblib.load('models/landslide_size_label_encoder.pkl')
trigger_encoder = joblib.load('models/landslide_trigger_label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            
            lat = float(request.form['latitude'])
            lon = float(request.form['longitude'])
            size = request.form['landslide_size']
            trigger = request.form['landslide_trigger']
            
            
            size_encoded = size_encoder.transform([size])[0]
            trigger_encoded = trigger_encoder.transform([trigger])[0]
            
           
            features = np.array([[lat, lon, size_encoded, trigger_encoded]])
            
           
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]  # Probability of 'high' impact
            
           
            m = folium.Map(location=[lat, lon], zoom_start=10)
            
           
            folium.Marker(
                [lat, lon],
                popup=f"Predicted Impact: {'High' if prediction == 1 else 'Low'}<br>Probability: {probability:.2%}",
                icon=folium.Icon(color='red' if prediction == 1 else 'green')
            ).add_to(m)
            
           
            map_html = m._repr_html_()
            
            return render_template('results.html', 
                                 prediction=prediction,
                                 probability=probability,
                                 map_html=map_html)
            
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)