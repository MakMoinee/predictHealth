import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the CSV data
df = pd.read_csv("formatted_health_data.csv")

# Display the first few rows to understand the structure of the data
print(df.head())

# Handle missing values by filling with zeros or using an imputer (based on your data)
df.fillna(0, inplace=True)

# Feature engineering: Let's say we want to predict 'heartRate' as an example.
# We'll use the other health metrics as features (you can modify this based on your goals).
X = df.drop(columns=["heartRate", "type"])  # Drop target column and 'type' if not needed for prediction
y = df["heartRate"]

# Convert categorical data into numerical (if necessary)
# If 'type' or any other column is categorical, you can apply label encoding
# This step is optional based on your feature set

# If 'type' is relevant, you can encode it:
if "type" in X.columns:
    label_encoder = LabelEncoder()
    X["type"] = label_encoder.fit_transform(X["type"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DMatrix (XGBoost's internal data format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters (these can be tuned for better performance)
params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',  # Root mean squared error as the evaluation metric
    'eta': 0.1,  # Learning rate
    'max_depth': 5,  # Max depth of the trees
    'subsample': 0.8,  # Fraction of samples used per tree
    'colsample_bytree': 0.8,  # Fraction of features used per tree
}

# Train the model using XGBoost's train function
num_round = 100  # Number of boosting rounds
bst = xgb.train(params, dtrain, num_round)

# Make predictions on the test set
y_pred = bst.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Optionally, save the trained model
bst.save_model("xgboost_model.json")
