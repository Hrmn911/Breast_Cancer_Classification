
import pandas as pd

# Load dataset
df = pd.read_csv("data/Breast_Cancer.csv")

# Display first five records
print("\nFIRST FIVE RECORDS")
print(df.head())

# Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Column information
print("\nDATASET INFORMATION")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Target distribution
print("\nSTATUS DISTRIBUTION")
print(df["Status"].value_counts())
