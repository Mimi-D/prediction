#imports
import json
from flask import Flask, jsonify, request
import joblib
from sklearn.ensemble import RandomForestClassifier

model = joblib.load('trained_rf_model.pkl')

app = Flask(__name__)

@app.route('/detect_leak')
def home():
    return "Prediction API, Model trained with random forest"
    
def preprocess_data(data):
    # Extract the features from the JSON data
    features = [data['pressure'], data['flowrate'], data['volume'], data['n1'], data['n2'], data['n3']]
    # Convert the features to a 2D list
    features_2d = [features]
    return features_2d

@app.route('/detect_leak', methods=['POST'])
def detect_leak():
    print(request.method)
    data = request.get_json()
    preprocessed_data = preprocess_data(data)
    prediction = model.predict(preprocessed_data)[0]
    # convert the numpy array to a list
    prediction = prediction.tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()
