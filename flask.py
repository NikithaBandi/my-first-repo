!pip install flask pandas scikit-learn tensorflow
from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mental_health_diagnosis_model.h5")

# Initialize Flask application
app = Flask(__name__)

# Define prediction function
def predict(input_data):
    # Preprocess input data
    X_new = pd.get_dummies(pd.DataFrame(input_data, index=[0]))
    # Make prediction
    prediction = model.predict(X_new)
    return prediction[0][0]

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    prediction = predict(data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
