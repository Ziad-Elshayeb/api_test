from flask import Flask, request, jsonify
import joblib
import numpy as np


# Load the trained model
model = joblib.load("model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "ML Model API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Ensure request is JSON
        if data is None:
            return jsonify({"error": "Empty request body"}), 400

        # Extract features
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        predicted_class = int(prediction[0])

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
