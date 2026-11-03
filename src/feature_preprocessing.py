from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv("data/Breast_Cancer_Cleaned.csv")

# Separate input features and target
X = df.drop(columns=["Status", "Survival Months"])
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Standardize Grade values
X["Grade"] = X["Grade"].replace({
    "anaplastic; Grade IV": "4"
})

# Numerical features
numerical_features = [
    "Age",
    "Tumor Size",
    "Regional Node Examined",
    "Regional Node Positive"
]

# Nominal categorical features
nominal_features = [
    "Race",
    "Marital Status",
    "A Stage",
    "Estrogen Status",
    "Progesterone Status"
]

# Ordinal categorical features
ordinal_features = [
    "T Stage",
    "N Stage",
    "6th Stage",
    "differentiate",
    "Grade"
]

ordinal_categories = [
    ["T1", "T2", "T3", "T4"],

    ["N1", "N2", "N3"],

    ["IIA", "IIB", "IIIA", "IIIB", "IIIC"],

    [
        "Well differentiated",
        "Moderately differentiated",
        "Poorly differentiated",
        "Undifferentiated"
    ],

    ["1", "2", "3", "4"]
]

# One-hot encoding pipeline
nominal_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# Ordinal encoding pipeline
ordinal_transformer = OrdinalEncoder(
    categories=ordinal_categories
)

# Combine all transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("nominal", nominal_transformer, nominal_features),
        ("ordinal", ordinal_transformer, ordinal_features)
    ],
    remainder="passthrough"
)

print("\nPreprocessing transformer created successfully!")

print("\nGrade values after standardization:")
print(X["Grade"].value_counts())

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nInput columns:")
print(X.columns.tolist())

print("\nTarget values:")
print(y.value_counts())

print("\nFeature data types:")
print(X.dtypes)

print("\nUnique values in categorical columns:")

categorical_columns = X.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(X[column].unique())


print("\nNumerical features:", numerical_features)
print("Nominal features:", nominal_features)
print("Ordinal features:", ordinal_features)

for feature, categories in zip(ordinal_features, ordinal_categories):
    print(f"{feature}: {categories}")

X_transformed = preprocessor.fit_transform(X)

print("\nOriginal feature shape:", X.shape)
print("Transformed feature shape:", X_transformed.shape)