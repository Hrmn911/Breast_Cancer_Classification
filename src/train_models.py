import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler

)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/Breast_Cancer_Cleaned.csv")

df["Grade"] = df["Grade"].replace({
    "anaplastic; Grade IV": "4"
})

X = df.drop(columns=["Status", "Survival Months"])
y = df["Status"]

# First split: training + test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: training + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

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

# Define category order
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

preprocessor = ColumnTransformer(
    transformers=[
        (
            "nominal",
            OneHotEncoder(handle_unknown="ignore"),
            nominal_features
        ),
        (
            "ordinal",
            OrdinalEncoder(categories=ordinal_categories),
            ordinal_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

balanced_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)

balanced_lr.fit(X_train, y_train)

# Get probability of each record belonging to Dead class
dead_index = list(
    balanced_lr.named_steps["classifier"].classes_
).index("Dead")

y_val_probabilities = balanced_lr.predict_proba(X_val)[:, dead_index]

print("\nFirst 10 Dead probabilities:")
print(y_val_probabilities[:10])

# Train model
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression trained successfully!")

y_pred = logistic_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=logistic_model.classes_
))


# Decision Tree Pipeline
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=42
        ))
    ]
)

# Train model
decision_tree_model.fit(X_train, y_train)

# Predictions
dt_predictions = decision_tree_model.predict(X_test)

# Evaluation
print("\nDECISION TREE RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test, dt_predictions)
)

print("\nClassification Report:")
print(
    classification_report(y_test, dt_predictions)
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        dt_predictions,
        labels=decision_tree_model.classes_
    )
)

# Random Forest Pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Train model
random_forest_model.fit(X_train, y_train)

# Predictions
rf_predictions = random_forest_model.predict(X_test)

# Evaluation
print("\nRANDOM FOREST RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test, rf_predictions)
)

print("\nClassification Report:")
print(
    classification_report(y_test, rf_predictions)
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions,
        labels=random_forest_model.classes_
    )
)

# KNN Pipeline
knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]
)

# Train model
knn_model.fit(X_train, y_train)

# Predictions
knn_predictions = knn_model.predict(X_test)

# Evaluation
print("\nKNN RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test, knn_predictions)
)

print("\nClassification Report:")
print(
    classification_report(y_test, knn_predictions)
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        knn_predictions,
        labels=knn_model.classes_
    )
)

# ============================================================
# CLASS IMBALANCE EXPERIMENT
# ============================================================

balanced_models = {
    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Balanced Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Balanced Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}


for model_name, classifier in balanced_models.items():

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\n{model_name.upper()}")

    print(
        "Accuracy:",
        accuracy_score(y_test, predictions)
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )

thresholds = [0.30, 0.40, 0.50, 0.60]

print("\nTHRESHOLD COMPARISON")

for threshold in thresholds:

    predictions = [
        "Dead" if probability >= threshold else "Alive"
        for probability in y_val_probabilities
    ]

    print(f"\nThreshold: {threshold}")

    print(
        "Precision:",
        precision_score(
            y_val,
            predictions,
            pos_label="Dead"
        )
    )

    print(
        "Recall:",
        recall_score(
            y_val,
            predictions,
            pos_label="Dead"
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_val,
            predictions,
            pos_label="Dead"
        )
    )

# ============================================================
# FINAL MODEL EVALUATION
# ============================================================

selected_threshold = 0.50

# Get probabilities on untouched test data
y_test_probabilities = balanced_lr.predict_proba(X_test)[:, dead_index]

# Apply selected threshold
final_predictions = [
    "Dead" if probability >= selected_threshold else "Alive"
    for probability in y_test_probabilities
]

print("\nFINAL BALANCED LOGISTIC REGRESSION")
print("Selected Threshold:", selected_threshold)

print("\nAccuracy:")
print(accuracy_score(y_test, final_predictions))

print("\nClassification Report:")
print(classification_report(y_test, final_predictions))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        final_predictions,
        labels=["Alive", "Dead"]
    )
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_artifact = {
    "model": balanced_lr,
    "threshold": selected_threshold,
    "target_class": "Dead"
}

joblib.dump(
    model_artifact,
    "models/balanced_logistic_regression.pkl"
)

print("\nFinal model saved successfully!")