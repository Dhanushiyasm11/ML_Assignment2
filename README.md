
# Dry Bean Classification Using Machine Learning

## Project Overview
This project focuses on classifying dry bean varieties using machine learning algorithms based on their physical and geometrical characteristics.

The dataset contains 16 independent features and 7 target classes.

## Dataset
- Dataset: Dry Bean Dataset
- Original dataset: Data/Dry_Bean_Dataset.xlsx
- Test dataset: test_data.csv
- Target variable: Class

## Machine Learning Algorithms
The following classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Model Evaluation
The models were evaluated using:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
ML_assignment2/
│
├── Data/
├── Models/
├── Notebooks/
├── app.py
├── model_comparison.csv
├── requirements.txt
├── test_data.csv
├── README.md
└── .gitignore
```

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Application Features
- Dataset overview
- Model selection
- Model performance comparison
- Confusion matrix
- Classification report
- Actual vs predicted results
- Downloadable prediction results
