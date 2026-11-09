
# ============================================================
# DRY BEAN CLASSIFICATION - STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification System")

st.markdown("""
An interactive machine learning dashboard for classifying
dry bean varieties using five trained classification models.
""")

st.divider()


# ============================================================
# 3. LOAD PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return scaler, label_encoder, feature_names


try:
    scaler, label_encoder, feature_names = load_artifacts()

except Exception as error:
    st.error(f"Error loading preprocessing files: {error}")
    st.stop()


# ============================================================
# 4. HANDLE STRING AND NUMERICAL CLASS LABELS
# ============================================================

def encode_labels(values):
    """
    Converts string class labels or already encoded numerical
    labels into the numerical format used by the LabelEncoder.
    """

    values = np.asarray(values).reshape(-1)

    # First, try converting original string class names.
    try:
        return label_encoder.transform(values.astype(str))

    except (ValueError, TypeError):
        pass

    # If labels are already numerical, retain their encoded values.
    try:
        numerical_values = values.astype(int)

        valid_classes = np.arange(len(label_encoder.classes_))

        if np.isin(numerical_values, valid_classes).all():
            return numerical_values

    except (ValueError, TypeError):
        pass

    raise ValueError("Unknown or invalid class labels detected.")


# ============================================================
# 5. TEST DATASET UPLOAD
# ============================================================

st.header("1. Test Dataset")

st.write("Upload your test dataset in CSV format.")

uploaded_file = st.file_uploader(
    "Choose a test CSV file",
    type=["csv"]
)

# Use uploaded CSV or automatically load the default test dataset.

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded test dataset loaded successfully!")

    except Exception as error:
        st.error(f"Error reading uploaded CSV: {error}")
        st.stop()

else:

    if TEST_DATA_PATH.exists():

        try:
            df = pd.read_csv(TEST_DATA_PATH)

            st.info(
                "Using the default test_data.csv from the project folder."
            )

        except Exception as error:
            st.error(f"Error reading default test dataset: {error}")
            st.stop()

    else:
        st.warning("Please upload a test CSV file to continue.")
        st.stop()


# ============================================================
# 6. VALIDATE AND PREPARE DATASET
# ============================================================

st.header("2. Dataset Overview")

# Check whether the target column exists.

if "Class" not in df.columns:
    st.error("Target column 'Class' is missing.")
    st.stop()

# Check whether all required features are available.

missing_features = [
    feature for feature in feature_names
    if feature not in df.columns
]

if missing_features:
    st.error(f"Missing features: {missing_features}")
    st.stop()

# Check for missing values.

if df[list(feature_names)].isnull().any().any():
    st.error("Missing values found in the input features.")
    st.stop()

# Check whether features contain numerical values.

if not all(
    pd.api.types.is_numeric_dtype(df[col])
    for col in feature_names
):
    st.error("All input features must be numerical.")
    st.stop()

# Separate independent features and target variable.

X_test = df[list(feature_names)].copy()

try:
    y_test = encode_labels(df["Class"])

except ValueError as error:
    st.error(str(error))
    st.stop()


# Display dataset summary.

col1, col2, col3 = st.columns(3)

col1.metric("Total Test Records", len(df))
col2.metric("Total Features", len(feature_names))
col3.metric("Number of Classes", df["Class"].nunique())

with st.expander("View Test Dataset"):
    st.dataframe(df.head(10), use_container_width=True)


# ============================================================
# 7. FEATURE SCALING
# ============================================================

try:

    # Apply the same scaler used during model training.

    X_test_scaled = scaler.transform(X_test)

except Exception as error:
    st.error(f"Feature scaling failed: {error}")
    st.stop()


# ============================================================
# 8. LOAD ALL FIVE TRAINED MODELS
# ============================================================

model_files = {

    "Logistic Regression": "logistic_regression.pkl",

    "Decision Tree": "decision_tree.pkl",

    "K-Nearest Neighbors": "knn.pkl",

    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",

    "Support Vector Machine": "svm.pkl"

}


@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


models = {}

for model_name, file_name in model_files.items():

    model_path = MODEL_DIR / file_name

    if not model_path.exists():
        st.error(f"Model file not found: {file_name}")
        st.stop()

    try:

        models[model_name] = load_model(str(model_path))

    except Exception as error:

        st.error(f"Unable to load {model_name}: {error}")
        st.stop()


# ============================================================
# 9. CALCULATE AUC SCORE
# ============================================================

def calculate_auc(model, X, y_true):

    try:

        # AUC requires probability scores for multiclass evaluation.

        if not hasattr(model, "predict_proba"):
            return None

        probabilities = model.predict_proba(X)

        # Convert model classes to encoded numerical labels.

        encoded_classes = encode_labels(model.classes_)

        # Arrange probability columns according to LabelEncoder order.

        total_classes = len(label_encoder.classes_)

        aligned_probabilities = np.zeros(
            (len(y_true), total_classes)
        )

        for index, class_id in enumerate(encoded_classes):

            aligned_probabilities[:, class_id] = probabilities[:, index]

        return roc_auc_score(
            y_true,
            aligned_probabilities,
            multi_class="ovr",
            average="weighted",
            labels=np.arange(total_classes)
        )

    except (ValueError, TypeError, IndexError):
        return None


# ============================================================
# 10. EVALUATE ALL FIVE MODELS
# ============================================================

st.header("3. Model Performance Comparison")

evaluation_results = {}
all_predictions = {}

for model_name, model in models.items():

    try:

        # Generate predictions using the trained model.

        raw_predictions = model.predict(X_test_scaled)

        # Handle both string and numerical predictions.

        predictions = encode_labels(raw_predictions)

        # Calculate evaluation metrics.

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        auc = calculate_auc(
            model,
            X_test_scaled,
            y_test
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            predictions
        )

        # Store the evaluation results.

        evaluation_results[model_name] = {

            "Accuracy": accuracy,
            "AUC Score": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC Score": mcc

        }

        all_predictions[model_name] = predictions

    except Exception as error:

        st.error(
            f"Evaluation failed for {model_name}: {error}"
        )

        st.stop()


# ============================================================
# 11. DISPLAY MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    evaluation_results
).T

comparison_df.index.name = "ML Model"

st.subheader("Comparison of All Five Models")

st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)

# Identify the best model based on accuracy.

best_model = comparison_df["Accuracy"].idxmax()

st.success(
    f"Best Performing Model: {best_model}"
)


# ============================================================
# 12. SELECT MODEL FOR DETAILED EVALUATION
# ============================================================

st.divider()

st.header("4. Detailed Model Evaluation")

selected_name = st.selectbox(
    "Select a Classification Model",
    list(models.keys())
)

selected_model = models[selected_name]

predictions = all_predictions[selected_name]


# ============================================================
# 13. DISPLAY PERFORMANCE METRICS
# ============================================================

st.subheader(f"Performance Metrics - {selected_name}")

metrics = evaluation_results[selected_name]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{metrics['Accuracy']:.4f}"
)

col2.metric(
    "AUC Score",
    f"{metrics['AUC Score']:.4f}"
    if metrics["AUC Score"] is not None else "N/A"
)

col3.metric(
    "Precision",
    f"{metrics['Precision']:.4f}"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Recall",
    f"{metrics['Recall']:.4f}"
)

col5.metric(
    "F1 Score",
    f"{metrics['F1 Score']:.4f}"
)

col6.metric(
    "MCC Score",
    f"{metrics['MCC Score']:.4f}"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

st.subheader("Confusion Matrix")

class_labels = label_encoder.classes_

cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(class_labels))
)

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_title(
    f"Confusion Matrix - {selected_name}"
)

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(class_labels)),
    target_names=class_labels,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# 16. ACTUAL VS PREDICTED RESULTS
# ============================================================

st.subheader("Actual vs Predicted Results")

# Convert encoded predictions back to original bean names.

predicted_labels = label_encoder.inverse_transform(
    predictions
)

results_df = pd.DataFrame({

    "Actual Class": df["Class"].astype(str).reset_index(drop=True),

    "Predicted Class": predicted_labels

})

# Identify correct and incorrect predictions.

results_df["Prediction Status"] = (

    results_df["Actual Class"] == results_df["Predicted Class"]

).map({

    True: "Correct",
    False: "Incorrect"

})

st.dataframe(
    results_df.head(20),
    use_container_width=True
)


# ============================================================
# 17. DOWNLOAD PREDICTION RESULTS
# ============================================================

st.subheader("Download Prediction Results")

csv_data = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="Download Prediction Results",

    data=csv_data,

    file_name="dry_bean_predictions.csv",

    mime="text/csv"

)


# ============================================================
# 18. FINAL STATUS
# ============================================================

st.divider()

st.success(
    "Dry Bean Classification and Model Evaluation Completed Successfully!"
)

st.caption(
    "All five trained classification models were evaluated "
    "using the supplied test dataset. No model retraining was performed."
)


# ============================================================
# END OF APPLICATION
# ============================================================
