import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

# Initialize label encoder (assuming you have a file with label classes)
label_encoder = LabelEncoder()
label_encoder.fit(["dehydration", "overfatigue", "heat stroke risk"])

# API route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (JSON)
        input_data = request.get_json()

        # Extract health data
        health_data = input_data.get("health_data", {})

        # Convert the health data into a pandas DataFrame (1 row, multiple features)
        df = pd.DataFrame([health_data])

        # Handle missing values by filling with 0 or more sophisticated methods
        df.fillna(0, inplace=True)

        # Make prediction (get both class labels and probabilities)
        probabilities = model.predict_proba(df)

        # Check if the number of classes matches the expected label encoder classes
        num_classes = probabilities.shape[1]
        print(f"Model predicts {num_classes} classes.")
        print(f"Label encoder has {len(label_encoder.classes_)} classes.")

        if num_classes != len(label_encoder.classes_):
            return jsonify({"error": f"Mismatch between number of classes in model ({num_classes}) and label encoder ({len(label_encoder.classes_)})"}), 400
        
        predicted_class = label_encoder.inverse_transform([probabilities.argmax(axis=1)[0]])[0]

        # Log the probabilities for inspection
        print(f"Probabilities: {probabilities}")

        # Return the prediction and probabilities as a JSON response
        response = {
            "prediction": predicted_class,
            "message": "Prediction successful"
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    try:
        response = {"healthy": True}
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
