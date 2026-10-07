import pandas as pd

data = pd.read_csv("data/air_quality_raw.csv")

print("Shape:", data.shape)
print("Columns:")
print(data.columns.tolist())

print("\nFirst five rows:")
print(data.head())
