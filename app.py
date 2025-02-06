from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained ML model
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with a key "data":
      {
        "data": [sepal_length, sepal_width, petal_length, petal_width]
      }
    Returns the model's prediction as JSON.
    """
    data = request.get_json()
    if not data or 'data' not in data:
        return jsonify({"error": "Invalid input. Expecting JSON with a 'data' key."}), 400

    input_data = data['data']
    try:
        # The model expects a 2D array: one sample per row.
        prediction = model.predict([input_data])
        # Convert the prediction to a standard Python type (e.g., int)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000 (debug mode enabled for development)
    app.run(host='0.0.0.0', port=5000, debug=True)
