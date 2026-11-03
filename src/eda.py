
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/Breast_Cancer_Cleaned.csv")

sns.set_theme(style="whitegrid")


# 1. Target distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Status"
)

plt.title("Breast Cancer Survival Status Distribution")
plt.xlabel("Survival Status")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# 2. Age distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 3. Tumour size distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Tumor Size",
    bins=25,
    kde=True
)

plt.title("Tumour Size Distribution")
plt.xlabel("Tumour Size")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 4. Cancer stage distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="6th Stage",
    hue="Status"
)

plt.title("Cancer Stage vs Survival Status")
plt.xlabel("Cancer Stage")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()



# Categorical feature analysis

categorical_columns = [
    "Race",
    "Marital Status",
    "T Stage",
    "N Stage",
    "differentiate",
    "Grade",
    "A Stage",
    "Estrogen Status",
    "Progesterone Status"
]

for column in categorical_columns:

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x=column,
        hue="Status"
    )

    plt.title(f"{column} vs Survival Status")
    plt.xlabel(column)
    plt.ylabel("Number of Records")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()


# Numerical correlation analysis

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numerical_columns.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()
plt.show()

