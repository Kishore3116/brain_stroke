# Brain Stroke Prediction Model

## Overview
This repository contains the code for building a **Brain Stroke Prediction Model** using machine learning techniques. The model predicts whether a person is at risk of having a stroke based on various health attributes such as age, heart disease, glucose level, BMI, smoking status, gender, marital status, and more.

The model uses a **Random Forest Classifier** to make predictions based on the dataset and evaluates its performance using accuracy and other metrics.

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn
- Matplotlib (optional for visualization)

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```
## Dataset
The dataset is a CSV file containing various features related to a patient's health and lifestyle, which are used to predict whether they are at risk for a brain stroke. The dataset includes columns such as:

- **age**: The age of the patient
- **gender**: The gender of the patient
- **hypertension**: Whether the patient has hypertension
- **heart_disease**: Whether the patient has heart disease
- **ever_married**: Whether the patient has ever been married
- **work_type**: The type of work the patient does
- **Residence_type**: The type of residence of the patient
- **avg_glucose_level**: Average glucose level of the patient
- **bmi**: Body Mass Index (BMI)
- **smoking_status**: Smoking status of the patient

## Steps for Building the Model

### 1. Data Preprocessing
- **Handling Missing Values**: Missing data is identified and handled.
- **Label Encoding**: Categorical columns such as `smoking_status`, `gender`, `ever_married`, `work_type`, and `Residence_type` are encoded using LabelEncoder.
- **Feature Selection**: Features such as `age`, `heart_disease`, `avg_glucose_level`, `bmi`, and `smoking_status` are selected for model training.

### 2. Train-Test Split
The dataset is split into training and testing sets using an 80-20 split ratio, ensuring that 80% of the data is used for training and 20% for testing.

### 3. Model Creation
A **Random Forest Classifier** is trained using the training data, and predictions are made using the test set.

### 4. Model Evaluation
The performance of the model is evaluated using the following metrics:
- **Accuracy Score**: The accuracy of the predictions.
- **Confusion Matrix**: Shows the number of correct and incorrect predictions.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

### 5. Prediction
The model is used to make predictions on a new data point.

Example prediction input:

```python
prediction = np.array([[79.0, 0, 174.12, 24.0, 2]])
prediction_df = pd.DataFrame(prediction, columns=xtrain.columns)
print(rfc.predict(prediction_df))
```
### 6. Result
The model predicts the likelihood of a brain stroke based on the input features.

## Output
The script will output:

- The accuracy of the Random Forest Classifier model.
- The confusion matrix and classification report.
- Predictions for the provided input data.

## Future Work
- Explore other machine learning models (e.g., Logistic Regression, SVM) to compare performance.
- Tune model hyperparameters for improved performance.
- Extend the dataset with more features or larger samples for better predictions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

