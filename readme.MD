Here is the README file for your Flask API using virtualenv:

---

# Health Prediction API

This API predicts the health condition (Dehydration, Overfatigue, or Heat Stroke Risk) based on the provided health data.

## Setup

Follow these steps to set up the project and run the Flask API in a virtual environment.

### Prerequisites

- Python 3.x
- Virtualenv

### Installation Steps

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/macOS
   venv\Scripts\activate      # On Windows
   ```

2. **Install dependencies:**

   After activating the virtual environment, install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   You can also install the necessary libraries manually:

   ```bash
   pip install flask xgboost scikit-learn pandas
   ```

3. **Start the Flask API:**

   Run the Flask API server with the following command:

   ```bash
   python app.py
   ```

   By default, the server will run on `http://127.0.0.1:5000`.

### API Endpoint

The API exposes a single POST endpoint:

**POST** `/predict`

#### Request Body
The request should be a JSON object containing two fields:
- `health_data`: A dictionary with the required health data.
- `type`: A string representing the health condition type (either "dehydration", "overfatigue", or "heat stroke risk").

Example request body:

```json
{
    "health_data": {
        "heartRate": 75,
        "restingHeartRate": 65,
        "walkingHeartRateAverage": 80,
        "activeEnergyBurned": 200,
        "basalEnergyBurned": 1500,
        "stepCount": 5000,
        "distanceWalkingRunning": 3.5
    },
    "type": "dehydration"
}
```

#### Response

The response will be a JSON object containing the predicted health condition.

Example response:

```json
{
    "prediction": "dehydration"
}
```

### Example Curl Command

You can test the API using the following `curl` command:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"health_data": {"heartRate": 75, "restingHeartRate": 65, "walkingHeartRateAverage": 80, "activeEnergyBurned": 200, "basalEnergyBurned": 1500, "stepCount": 5000, "distanceWalkingRunning": 3.5}, "type": "dehydration"}'
```

This will return the prediction based on the provided data.

### Troubleshooting

- Ensure that your virtual environment is activated before installing dependencies and running the server.
- If you encounter any issues with package versions, make sure you are using compatible versions listed in the `requirements.txt` file.


--- 