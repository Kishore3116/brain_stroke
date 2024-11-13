Brain Stroke Prediction with Random Forest Classifier
This repository contains a machine learning project for predicting brain stroke using various health-related features. The model is built using the Random Forest Classifier algorithm, which is a powerful ensemble learning technique. The dataset contains several attributes like age, heart disease, average glucose level, BMI, smoking status, and more, to predict the likelihood of a stroke.

Table of Contents
Project Overview
Dataset
Features
Model
Installation
Usage
Results
License
Project Overview
In this project, we focus on building a machine learning model that predicts whether a person is at risk of having a brain stroke based on various health-related features. The prediction is based on data collected from a variety of patients and is built to help in early detection, which is crucial in preventing strokes. We use the Random Forest Classifier to train the model and evaluate its performance.

Dataset
The dataset used in this project is publicly available and contains the following columns:

age: Age of the person.
gender: Gender of the person (binary encoded: 0 for female, 1 for male).
hypertension: Whether the person suffers from hypertension (0: No, 1: Yes).
heart_disease: Whether the person has heart disease (0: No, 1: Yes).
ever_married: Whether the person has ever been married (binary encoded).
work_type: Type of employment (binary encoded).
Residence_type: Type of residence (binary encoded).
avg_glucose_level: Average glucose level of the person.
bmi: Body Mass Index.
smoking_status: Smoking status of the person (binary encoded).
stroke: Target variable indicating if the person had a stroke (1: Yes, 0: No).
You can find the dataset in the brain_stroke.csv file.

Features
The following features are used to train the model:

Age
Heart Disease
Average Glucose Level
BMI
Smoking Status
The target variable is whether the person had a stroke or not.

Model
We use the Random Forest Classifier, an ensemble method that creates a collection of decision trees and combines their outputs for better predictions. This approach is known for its high accuracy and robustness against overfitting.

Steps involved in the project:
Data Preprocessing:

Handle missing data (if any).
Convert categorical data (like gender, smoking_status, etc.) into numerical values using Label Encoding.
Feature Extraction:

Select relevant features for the model (e.g., age, heart disease, average glucose level, etc.).
Model Creation:

Split the data into training and test sets.
Train the Random Forest Classifier on the training data.
Model Evaluation:

Use metrics like accuracy, classification report, and confusion matrix to evaluate the model.
Prediction:

Make predictions on new instances of data using the trained model.
Installation
To get started with the project, follow these steps:

Prerequisites
Make sure you have Python installed on your system. It is recommended to create a virtual environment for your project.

Install the necessary libraries using pip:
bash
Copy code
pip install numpy pandas scikit-learn matplotlib
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/your-username/brain-stroke-prediction.git
cd brain-stroke-prediction
Usage
Load the dataset: The dataset is in brain_stroke.csv. Load it using pandas.

Preprocess the data: Handle missing values and encode categorical features.

Train the model: Use Random Forest Classifier to train the model.

Evaluate the model: Check the model's performance using accuracy, classification report, and confusion matrix.

Make Predictions: Use the trained model to make predictions for new data instances.

Here’s how to run the model:

python
Copy code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing

# Load dataset
df = pd.read_csv("path/to/brain_stroke.csv")

# Data Preprocessing
df['smoking_status'] = preprocessing.LabelEncoder().fit_transform(df['smoking_status'])
df['gender'] = preprocessing.LabelEncoder().fit_transform(df['gender'])
df['ever_married'] = preprocessing.LabelEncoder().fit_transform(df['ever_married'])
df['work_type'] = preprocessing.LabelEncoder().fit_transform(df['work_type'])
df['Residence_type'] = preprocessing.LabelEncoder().fit_transform(df['Residence_type'])

# Feature extraction
X = df[["age", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"]]
y = df['stroke']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(rfc.predict(X_test), y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(y_test, rfc.predict(X_test)))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rfc.predict(X_test)))

# Make prediction for a new instance
prediction = np.array([[79.0, 0, 174.12, 24.0, 2]])  # Example data for prediction
prediction_df = pd.DataFrame(prediction, columns=X_train.columns)
predicted_value = rfc.predict(prediction_df)
print(f"\nPrediction for input data: {predicted_value}")
Results
The model achieves an accuracy of XX% on the test data, and the confusion matrix shows how well the model is distinguishing between the classes (stroke vs no stroke).

Example output for classification report:
plaintext
Copy code
               precision    recall  f1-score   support

           0       0.96      0.98      0.97       982
           1       0.94      0.86      0.90       118

    accuracy                           0.96      1100
   macro avg       0.95      0.92      0.94      1100
weighted avg       0.96      0.96      0.96      1100
Example confusion matrix:
plaintext
Copy code
[[964  18]
 [ 17 101]]
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Thanks to Kaggle for providing the dataset.
Thanks to the authors of the scikit-learn library for providing easy-to-use tools for machine learning.
