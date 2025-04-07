import json
import pandas as pd

# Load raw data from JSON file
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize a list to store parsed entries
parsed_entries = []

# Iterate over the dictionary values
for key, values in data.items():
    # Check if the value is a list, and handle accordingly
    if isinstance(values, list) and len(values) > 0:
        # Get the last entry in the list, assuming it's an array of dicts
        for entry in values:
            try:
                parsed_entry = {
                    "heartRate": entry.get("value", 0) if key == "heartRate" else None,
                    "stepCount": entry.get("value", 0) if key == "stepCount" else None,
                    "distanceWalkingRunning": entry.get("value", 0) if key == "distanceWalkingRunning" else None,
                    "activeEnergyBurned": entry.get("value", 0) if key == "activeEnergyBurned" else None,
                    "restingHeartRate": entry.get("value", 0) if key == "restingHeartRate" else None,
                    "walkingHeartRateAverage": entry.get("value", 0) if key == "walkingHeartRateAverage" else None,
                    "type": key  # Store the key type if needed
                }
                parsed_entries.append(parsed_entry)
            except Exception as e:
                print(f"Error parsing entry: {e}")

# Convert parsed entries into a DataFrame
df = pd.DataFrame(parsed_entries)

# Save the DataFrame to CSV
if not df.empty:
    df.to_csv("formatted_health_data.csv", index=False)
    print("Saved formatted data to formatted_health_data.csv")
else:
    print("No valid health data to save.")
