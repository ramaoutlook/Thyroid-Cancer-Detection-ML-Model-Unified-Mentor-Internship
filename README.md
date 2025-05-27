# Thyroid Cancer Recurrence Prediction Project

This project focuses on building machine learning models to predict the recurrence of thyroid cancer based on patient data. The analysis involves data loading, exploration, visualization, preprocessing, and training of predictive models.

## Project Overview

Thyroid cancer is a common endocrine malignancy. Predicting its recurrence is crucial for patient management and follow-up. This project aims to leverage a dataset containing various patient attributes to train models that can classify whether a patient is likely to experience cancer recurrence.

## Dataset

The project uses the `thyroid_cancer_data.csv` dataset. It contains information about patients, including age, gender, smoking status, risk level, cancer stage, and a target variable indicating whether the cancer recurred (`Recurred`).

## Analysis and Methodology

The jupyter notebook (`thyroid_cancer_prediction.ipynb` - assuming this is the filename) follows a standard machine learning workflow:

1.  **Data Loading and Initial Exploration:**
    *   Loading the dataset using pandas.
    *   Checking the shape, head, information, and descriptive statistics of the data.
    *   Identifying and handling missing values (though the provided code shows no missing values).
    *   Examining the distribution of the target variable (`Recurred`).

2.  **Exploratory Data Analysis (EDA):**
    *   Visualizing the distribution of the target variable.
    *   Analyzing the relationship between features (like Age, Gender, Smoking, Risk, Stage) and the target variable using histograms, boxplots, and bar plots.
    *   Creating a correlation heatmap for numerical features to understand their relationships.

3.  **Data Preprocessing:**
    *   Separating features (`X`) and the target variable (`y`).
    *   Identifying categorical columns.
    *   Applying one-hot encoding to categorical features using `pd.get_dummies`.
    *   Encoding the target variable (`Recurred`) from categorical (Yes/No) to numerical (1/0) using `LabelEncoder`.

4.  **Data Splitting:**
    *   Splitting the preprocessed data into training and testing sets using `train_test_split` with a test size of 20% and stratification to maintain the target distribution.

5.  **Model Training and Evaluation:**
    *   Training two classification models:
        *   Random Forest Classifier
        *   Logistic Regression
    *   Making predictions on the test set.
    *   Evaluating model performance using:
        *   Accuracy Score
        *   Classification Report (Precision, Recall, F1-score)
        *   Confusion Matrix

6.  **Feature Importance (for Random Forest):**
    *   Analyzing the importance of different features in the Random Forest model to understand which factors contribute most to the prediction.

## Requirements

To run this notebook, you will need the following libraries:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `sklearn` (for `train_test_split`, `LabelEncoder`, `RandomForestClassifier`, `LogisticRegression`, `accuracy_score`, `classification_report`, `confusion_matrix`)

You can install them using pip:

Please see the visualizations of the project and some outputs
![Image](https://github.com/user-attachments/assets/d5cbcd7f-0326-4b8d-8d5a-fad5a8cc01c5)

![Image](https://github.com/user-attachments/assets/f1f49286-fc9f-4e9b-b4c7-e1896160fedb)

![Image](https://github.com/user-attachments/assets/aba091da-28d0-422f-ab74-b6460738bd76)


