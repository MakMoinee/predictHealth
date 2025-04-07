import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model("xgboost_model.json")

# Initialize Flask app
app = Flask(__name__)

# Assuming we have a label encoder for 'type'
label_encoder = LabelEncoder()
label_encoder.fit(["dehydration", "overfatigue", "heat stroke risk"])

# API route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (JSON)
        input_data = request.get_json()

        # Extract health data and the type
        health_data = input_data.get("health_data", {})
        target_type = input_data.get("type", "").lower()

        if target_type not in label_encoder.classes_:
            return jsonify({"error": f"Invalid type. Must be one of {label_encoder.classes_}"}), 400

        # Convert the health data into a pandas DataFrame
        df = pd.DataFrame([health_data])

        # Handle missing values by filling with 0 or more sophisticated methods
        df.fillna(0, inplace=True)

        # If you have a specific feature extraction method, use it here
        # For this example, we'll assume the columns are already correct for prediction
        X = df  # You can further process this if needed

        # Create a DMatrix (XGBoost internal format) for prediction
        dtest = xgb.DMatrix(X)

        # Get the prediction (model output)
        prediction = model.predict(dtest)

        # Map the prediction to the target type using the label encoder
        predicted_class = label_encoder.inverse_transform([int(prediction.argmax())])[0]

        # Return the prediction as a JSON response
        response = {
            "prediction": predicted_class,
            "message": "Prediction successful"
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
