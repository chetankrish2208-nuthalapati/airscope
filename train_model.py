import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data/air_quality_raw.csv")

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

features = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "Wind Speed"
]

for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=features).copy()

def calculate_pollution_score(row):
    pm25_score = min(row["PM2.5"] / 150, 1)
    pm10_score = min(row["PM10"] / 250, 1)
    no2_score = min(row["NO2"] / 200, 1)
    so2_score = min(row["SO2"] / 100, 1)
    co_score = min(row["CO"] / 15, 1)
    o3_score = min(row["O3"] / 200, 1)

    return (
        0.35 * pm25_score
        + 0.25 * pm10_score
        + 0.15 * no2_score
        + 0.10 * so2_score
        + 0.10 * co_score
        + 0.05 * o3_score
    )

data["PollutionScore"] = data.apply(calculate_pollution_score, axis=1)

def assign_risk(score):
    if score < 0.35:
        return "Low"
    elif score < 0.65:
        return "Moderate"
    else:
        return "High"

data["Risk"] = data["PollutionScore"].apply(assign_risk)

X = data[features]
y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Rows used:", len(data))
print("\nRisk distribution:")
print(data["Risk"].value_counts())

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "air_quality_model.joblib"
)

data.to_csv("data/air_quality_clean.csv", index=False)

print("\nSaved:")
print("- air_quality_model.joblib")
print("- data/air_quality_clean.csv")
