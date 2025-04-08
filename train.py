import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Step 1: Load your dataset
df = pd.read_csv('formatted_health_data.csv')

# Step 2: Preprocess - remove 'type' column
X = df.drop(columns=['type'])

# Step 3: Handle missing values (fill NaNs with column mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Optional: Save the imputer for use in predictions
import joblib
joblib.dump(imputer, "imputer.pkl")

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
joblib.dump(scaler, "scaler.pkl")

# Step 5: KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Step 6: Manual mapping of clusters to health statuses
cluster_to_health = {
    0: "normal",
    1: "dehydration",
    2: "overfatigue"
}
df['health_status'] = [cluster_to_health[label] for label in cluster_labels]

# Step 7: Encode labels and split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, df['health_status'], test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
joblib.dump(label_encoder, "label_encoder.pkl")

# Step 8: Train XGBoost Classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train_encoded)
model.save_model("xgboost_model.json")

# Step 9: Evaluate
y_pred = model.predict(X_test)
print(classification_report(
    y_test_encoded,
    y_pred,
    labels=label_encoder.transform(label_encoder.classes_),
    target_names=label_encoder.classes_
))

