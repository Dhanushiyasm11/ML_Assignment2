
# Dry Bean Classification Using Machine Learning

## Project Overview

This project focuses on classifying dry bean varieties using machine learning algorithms based on their physical and geometrical characteristics.

The dataset contains 16 independent features and 7 target classes. Five classification algorithms are implemented, trained, and evaluated to identify the best-performing model.

## Problem Statement

The objective of this project is to develop a machine learning-based classification system that can accurately identify different varieties of dry beans based on their physical and geometrical characteristics.

Multiple classification algorithms are evaluated using standard performance metrics to determine the most suitable model.

## Dataset

- **Dataset:** Dry Bean Dataset
- **Original dataset:** `Data/Dry_Bean_Dataset.xlsx`
- **Test dataset:** `test_data.csv`
- **Target variable:** `Class`
- **Number of independent features:** 16
- **Number of target classes:** 7

## Machine Learning Algorithms

The following classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Model Comparison Results

The following table presents the performance of all five machine learning models.

| Model                | Accuracy |      AUC | Precision |   Recall | F1 Score |      MCC |
| -------------------- | -------: | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression  | 0.921410 | 0.993408 |  0.922178 | 0.921410 | 0.921590 | 0.905045 |
| Decision Tree        | 0.891664 | 0.932962 |  0.891247 | 0.891664 | 0.891322 | 0.869059 |
| KNN                  | 0.916636 | 0.981247 |  0.917409 | 0.916636 | 0.916805 | 0.899204 |
| Gaussian Naive Bayes | 0.763863 | 0.964442 |  0.765371 | 0.763863 | 0.761474 | 0.715406 |
| SVM                  | 0.922145 | 0.993680 |  0.922426 | 0.922145 | 0.922177 | 0.905870 |

## Model Performance Observations

1. **Logistic Regression:** Achieved 92.14% accuracy and 0.993408 AUC, demonstrating strong classification performance.

2. **Decision Tree:** Achieved 89.17% accuracy, showing comparatively lower performance than Logistic Regression, KNN, and SVM.

3. **KNN:** Achieved 91.66% accuracy and 0.981247 AUC, demonstrating good predictive performance.

4. **Gaussian Naive Bayes:** Achieved 76.39% accuracy, the lowest among the five evaluated models.

5. **Support Vector Machine (SVM):** Achieved the highest accuracy of 92.21%, along with the highest AUC, Precision, Recall, F1 Score, and MCC.

## Overall Best Model

Based on the experimental results, Support Vector Machine (SVM) is identified as the best-performing classification algorithm among the five evaluated models.

Its performance metrics are:

- Accuracy: 92.21%
- AUC: 0.993680
- Precision: 0.922426
- Recall: 0.922145
- F1 Score: 0.922177
- MCC: 0.905870

Therefore, SVM demonstrates the best overall classification performance for the Dry Bean Dataset among the evaluated algorithms.

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
ML_Assignment2/
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

### 1. Clone the Repository

```bash
git clone https://github.com/Dhanushiyasm11/ML_Assignment2.git
cd ML_Assignment2
```

### 2. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

## Application Features

- Dataset overview
- Model selection
- Model performance comparison
- Confusion matrix
- Classification report
- Actual vs. predicted results
- Downloadable prediction results

## GitHub Repository

[Dry Bean Classificastion Using Machine Learning](https://github.com/Dhanushiyasm11/ML_Assignment2)
## Live Demo

The application is deployed using Streamlit Community Cloud.

**Live App:** https://mlassignment2-fcy3ko97texqqkfkjg3zay.streamlit.app/