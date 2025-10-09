import pandas as pd

# Load merged dataset
merged = pd.read_csv("merged_dataset.csv")

# List of pollutant columns
pollutants = ["co","no","no2","o3","so2","pm2_5","pm10","nh3"]

# Separate columns into weather and pollutants
all_cols = merged.columns.tolist()

# Ensure pollutants exist in merged file
weather_cols = [c for c in all_cols if c not in pollutants]

# Reorder: weather first, pollutants last
new_order = weather_cols + pollutants
merged = merged[new_order]

# Save new file
merged.to_csv("merged_dataset_reordered.csv", index=False)

print("Columns reordered! New file saved.")
print(merged.head())
