import pandas as pd
df1 = pd.read_csv("delhi_aqi.csv")  
df2 = pd.read_csv("open-meteo-28.65N77.27E231m.csv")

df1['date'] = pd.to_datetime(df1['date'], format="%d-%m-%Y %H:%M")
df2['time'] = pd.to_datetime(df2['time'], format="%Y-%m-%dT%H:%M")

df1 = df1.rename(columns={'date': 'datetime'})
df2 = df2.rename(columns={'time': 'datetime'})

merged = pd.merge(df1, df2, on="datetime", how="inner")

merged.to_csv("merged_dataset.csv", index=False)

print("Merged dataset shape:", merged.shape)
print(merged.head())
