# Brain Stroke Prediction Model

## Overview

This project aims to predict the likelihood of a brain stroke based on various health-related attributes using machine learning techniques. The model is built using a Decision Tree Classifier and is trained on a dataset containing information about patients' demographics and health conditions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you will need Python and the following libraries:

- `pandas`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn
```

## Data Preprocessing

The dataset used in this project is processed as follows:

- **Label Encoding**: Categorical variables such as `smoking_status`, `gender`, `ever_married`, `work_type`, and `Residence_type` are encoded into numerical values for model training.
- **Missing Values**: The dataset is checked for missing values, and appropriate handling is applied (if any).

## Model Training

The model is trained using the Decision Tree Classifier from the `scikit-learn` library. The dataset is split into training and testing sets using an 80-20 ratio.

### Evaluation Metrics

- **Accuracy**: The accuracy of the model is calculated on the test set.
- **Confusion Matrix**: A confusion matrix is generated to visualize the performance of the model.
- **Classification Report**: Detailed metrics including precision, recall, and F1-score are provided.

## Results

The model's performance is evaluated, and the results are displayed. An example prediction can be made with the trained model:

```python
prediction = dtc.predict([[0, 49.0, 0, 0, 1, 1, 1, 171.23, 34.4, 3]])
print(prediction)
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, feel free to open an issue or submit a pull request.

---
