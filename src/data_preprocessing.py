
import pandas as pd

# Load original dataset
df = pd.read_csv("dataset/Breast_Cancer.csv")

print("Original dataset shape:", df.shape)

# 1. Standardize column names
df.columns = df.columns.str.strip()

# Correct spelling for internal consistency
df = df.rename(columns={
    "Reginol Node Positive": "Regional Node Positive"
})

# 2. Remove unnecessary spaces from text values
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    df[column] = df[column].str.strip()

# 3. Remove duplicate records
df = df.drop_duplicates()

# 4. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Check duplicates
print("\nDuplicate records:", df.duplicated().sum())

# 6. Check target distribution
print("\nTarget distribution:")
print(df["Status"].value_counts())

# 7. Final dataset shape
print("\nCleaned dataset shape:", df.shape)

# Save cleaned dataset
df.to_csv(
    "dataset/Breast_Cancer_Cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
