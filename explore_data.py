import pandas as pd
# Load the dataset
df = pd.read_csv("data/Crop_recommendation.csv")
# Display the first 5 rows
print(df.head())
print("\n DATASET SHAPE ")
print(df.shape)

print("\nCOLUMN NAMES ")
print(df.columns)

print("\nDATA TYPES ")
print(df.dtypes)

print("\n MISSING VALUES ")
print(df.isnull().sum())

print("\n DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n CROP COUNTS ")
print(df["label"].value_counts())

print("\n STATISTICAL SUMMARY ")
print(df.describe())