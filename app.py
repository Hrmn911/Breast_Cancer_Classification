import pandas as pd
import streamlit as st
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/balanced_logistic_regression.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please train the model first.")
    st.stop()

model_artifact = joblib.load(MODEL_PATH)

model = model_artifact["model"]
threshold = model_artifact["threshold"]
target_class = model_artifact["target_class"]

# ============================================================
# LOAD CLEANED DATASET
# ============================================================

DATA_PATH = "data/Breast_Cancer_Cleaned.csv"

if not os.path.exists(DATA_PATH):
    st.error("Cleaned dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["Grade"] = df["Grade"].replace({
    "anaplastic; Grade IV": "4"
})

# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🎗️ Breast Cancer Survival Status Classification")

st.write(
    "An educational machine learning application "
    "that demonstrates classification using a public dataset."
)

st.divider()
st.subheader("About the Project")

st.write("""
This project uses machine learning to classify recorded breast
cancer survival status as Alive or Dead.
""")

st.info("""
**Educational Purpose Only:** This application is not a diagnostic
tool and must not be used for medical decisions.
""")

st.write("**Selected Model:** Balanced Logistic Regression")
# ============================================================
# APPLICATION NAVIGATION
# ============================================================

overview_tab, classification_tab, evaluation_tab = st.tabs([
    "📊 Overview",
    "🤖 Classification",
    "📈 Model Evaluation"
])



# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    st.divider()
    st.header("📊 Dataset Analytics Dashboard")

    st.write(
        "Explore the cleaned breast cancer dataset "
        "used in this educational classification project."
    )

    # --------------------------------------------------------
    # 1. KEY DATASET STATISTICS
    # --------------------------------------------------------

    total_records = len(df)

    alive_count = (df["Status"] == "Alive").sum()
    dead_count = (df["Status"] == "Dead").sum()

    alive_percentage = (alive_count / total_records) * 100
    dead_percentage = (dead_count / total_records) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total_records)
    col2.metric("Alive Records", alive_count)
    col3.metric("Dead Records", dead_count)
    col4.metric("Average Age", f"{df['Age'].mean():.1f}")

    st.divider()

    # --------------------------------------------------------
    # 2. SURVIVAL STATUS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Survival Status Distribution")

    status_data = df["Status"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(status_data)

    with col2:

        st.write("**Status percentages**")

        st.write(f"Alive: {alive_percentage:.2f}%")
        st.write(f"Dead: {dead_percentage:.2f}%")

        st.write(
            "The dataset contains more Alive records than Dead records. "
            "This class imbalance is important when evaluating the model."
        )

    st.divider()

    # --------------------------------------------------------
    # 3. AGE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Age Distribution")

    age_distribution = pd.cut(
        df["Age"],
        bins=8
    ).value_counts().sort_index()

    age_distribution.index = age_distribution.index.astype(str)

    st.bar_chart(age_distribution)

    st.divider()

    # --------------------------------------------------------
    # 4. TUMOUR SIZE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Tumour Size Distribution")

    tumor_distribution = pd.cut(
        df["Tumor Size"],
        bins=8
    ).value_counts().sort_index()

    tumor_distribution.index = tumor_distribution.index.astype(str)

    st.bar_chart(tumor_distribution)

    st.divider()

    # --------------------------------------------------------
    # 5. CANCER STAGE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Cancer Stage Distribution")

    stage_distribution = pd.crosstab(
        df["6th Stage"],
        df["Status"]
    )

    st.bar_chart(stage_distribution)

    st.caption(
        "All charts describe patterns in the dataset. "
        "They do not establish medical conclusions or individual prognosis."
    )


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

numerical_features = [
    "Age",
    "Tumor Size",
    "Regional Node Examined",
    "Regional Node Positive"
]

ordinal_options = {
    "T Stage": ["T1", "T2", "T3", "T4"],
    "N Stage": ["N1", "N2", "N3"],
    "6th Stage": ["IIA", "IIB", "IIIA", "IIIB", "IIIC"],
    "differentiate": [
        "Well differentiated",
        "Moderately differentiated",
        "Poorly differentiated",
        "Undifferentiated"
    ],
    "Grade": ["1", "2", "3", "4"]
}

nominal_features = [
    "Race",
    "Marital Status",
    "A Stage",
    "Estrogen Status",
    "Progesterone Status"
]


# ============================================================
# CLASSIFICATION TAB
# ============================================================

with classification_tab:

    st.divider()
    st.header("🤖 Breast Cancer Status Classification")

    st.write(
        "Enter values from the dataset to demonstrate "
        "how the trained model processes a record."
    )

    user_input = {}

    # --------------------------------------------------------
    # PREDICTION FORM
    # --------------------------------------------------------

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        all_features = (
            numerical_features
            + nominal_features
            + list(ordinal_options.keys())
        )

        for index, feature in enumerate(all_features):

            current_column = col1 if index % 2 == 0 else col2

            with current_column:

                if feature in numerical_features:

                    minimum = int(df[feature].min())
                    maximum = int(df[feature].max())
                    default = int(df[feature].median())

                    user_input[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1
                    )

                elif feature in ordinal_options:

                    user_input[feature] = st.selectbox(
                        feature,
                        options=ordinal_options[feature]
                    )

                else:

                    options = sorted(
                        df[feature].dropna().unique().tolist()
                    )

                    user_input[feature] = st.selectbox(
                        feature,
                        options=options
                    )

        submitted = st.form_submit_button(
            "Classify Record",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        input_df = pd.DataFrame([user_input])

        dead_index = list(
            model.named_steps["classifier"].classes_
        ).index(target_class)

        dead_score = model.predict_proba(input_df)[0][dead_index]

        if dead_score >= threshold:
            prediction = "Dead"
        else:
            prediction = "Alive"

        st.divider()
        st.subheader("Classification Result")

        if prediction == "Dead":
            st.warning("Predicted recorded status: Dead")
        else:
            st.success("Predicted recorded status: Alive")

        st.metric(
            "Model score for Dead class",
            f"{dead_score * 100:.2f}%"
        )

        st.caption(
            "This is a model-generated score, not a calibrated "
            "clinical probability or an individual's prognosis."
        )

        st.info(
            f"The classification used a decision threshold of {threshold:.2f}."
        )

        st.dataframe(
            input_df,
            use_container_width=True
        )

        st.warning(
            "Educational demonstration only. This output must not "
            "be used for diagnosis, treatment, or medical decisions."
        )


# ============================================================
# MODEL EVALUATION TAB
# ============================================================

with evaluation_tab:

    st.divider()
    st.header("📈 Model Evaluation")

    st.write(
        "Performance comparison and evaluation of the "
        "selected Balanced Logistic Regression model."
    )

    # --------------------------------------------------------
    # 1. MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    comparison_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors",
            "Balanced Logistic Regression",
            "Balanced Decision Tree",
            "Balanced Random Forest"
        ],
        "Accuracy": [
            0.8534, 0.7578, 0.8398, 0.8273,
            0.6994, 0.7714, 0.8360
        ],
        "Dead Precision": [
            0.58, 0.22, 0.43, 0.34,
            0.28, 0.24, 0.36
        ],
        "Dead Recall": [
            0.15, 0.24, 0.15, 0.14,
            0.61, 0.23, 0.10
        ],
        "Dead F1-Score": [
            0.23, 0.23, 0.23, 0.20,
            0.38, 0.23, 0.15
        ]
    })

    st.dataframe(
        comparison_data.style.format({
            "Accuracy": "{:.2%}",
            "Dead Precision": "{:.2%}",
            "Dead Recall": "{:.2%}",
            "Dead F1-Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Why Balanced Logistic Regression?")

    st.write("""
    Standard Logistic Regression achieved higher accuracy but
    identified only 15% of the Dead class.

    Balanced Logistic Regression achieved 61% recall and an
    F1-score of 0.38 for the Dead class. Therefore, it was
    selected for this educational demonstration because it
    identifies more records belonging to the minority class.
    """)

    st.divider()

    # --------------------------------------------------------
    # 2. OVERALL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Overall Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "69.94%")
    col2.metric("Macro Precision", "59.00%")
    col3.metric("Macro Recall", "66.00%")

    st.divider()

    # --------------------------------------------------------
    # 3. CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader("Classification Report")

    report_data = {
        "Class": [
            "Alive",
            "Dead",
            "Macro Average",
            "Weighted Average"
        ],
        "Precision": [0.91, 0.28, 0.59, 0.81],
        "Recall": [0.72, 0.61, 0.66, 0.70],
        "F1-Score": [0.80, 0.38, 0.59, 0.74],
        "Support": [682, 123, 805, 805]
    }

    report_df = pd.DataFrame(report_data)

    st.dataframe(
        report_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Values are rounded from the previously obtained "
        "test classification report."
    )

    st.divider()

    # --------------------------------------------------------
    # 4. CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        [
            [488, 194],
            [48, 75]
        ],
        index=["Actual Alive", "Actual Dead"],
        columns=["Predicted Alive", "Predicted Dead"]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 5. CONFUSION MATRIX EXPLANATION
    # --------------------------------------------------------

    st.write("**Understanding the results:**")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Correct Alive Predictions", 488)
        st.metric("Correct Dead Predictions", 75)

    with col2:
        st.metric("Alive Classified as Dead", 194)
        st.metric("Dead Classified as Alive", 48)

    st.divider()

    # --------------------------------------------------------
    # 6. MODEL OBSERVATIONS
    # --------------------------------------------------------

    st.subheader("Model Performance Observations")

    st.write("""
    - The model correctly classified 488 Alive records.
    - The model correctly classified 75 Dead records.
    - The recall for the Dead class was approximately 61%.
    - The Dead class has lower precision than the Alive class.
    - The model uses a decision threshold of 0.50.
    """)

    st.warning(
        "These are exploratory results from the previously shared "
        "test evaluation. The test set was also inspected during "
        "model comparisons, so these results should not be treated "
        "as a completely untouched final evaluation."
    )

    st.caption(
        "Educational machine learning evaluation only. "
        "These metrics do not establish clinical usefulness."
    )