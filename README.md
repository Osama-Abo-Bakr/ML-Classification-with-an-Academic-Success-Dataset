# Classification with an Academic Success Dataset

## Project Overview

This project aims to classify students based on academic success using a variety of machine learning algorithms. The dataset includes features related to student demographics and academic history. The project focuses on data preprocessing, visualization, model building, and evaluation to derive meaningful insights and predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Libraries and Tools](#libraries-and-tools)
- [Project Workflow](#project-workflow)
- [Model Evaluation](#model-evaluation)
- [Installation](#installation)
- [Contact](#contact)

## Libraries and Tools

- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms
- **Imbalanced-learn**: Handling imbalanced datasets
- **XGBoost**: Gradient boosting
- **FLAML**: Automated machine learning
- **AutoGluon**: Automated machine learning

## Project Workflow

1. **Reading and Exploring Data**
    - Loaded the dataset and displayed the first few records.
    - Checked data types and null values.

2. **Data Cleaning and Preprocessing**
    - Removed irrelevant columns.
    - Encoded categorical features using LabelEncoder.
    - Addressed class imbalance using SMOTE.
    - Normalized the features with MinMaxScaler.

3. **Data Visualization**
    - Plotted histograms to visualize feature distributions.
    - Displayed the distribution of target classes.

4. **Splitting Data**
    - Split the dataset into training and testing sets for model evaluation.

5. **Model Building and Evaluation**
    - Implemented and evaluated several classification models:
        - **Random Forest Classifier**
        - **AdaBoost Classifier**
        - **Gradient Boosting Classifier**
        - **XGBoost Classifier**
        - **Voting Classifier**
        - **AutoML (FLAML)**
        - **AutoGluon**
    - Evaluated models based on accuracy, precision, recall, mean squared error, mean absolute error, and classification report.

## Model Evaluation

### Random Forest Classifier
- **Training Accuracy**: 0.999
- **Testing Accuracy**: 0.85

### AdaBoost Classifier
- **Training Accuracy**: 1.0
- **Testing Accuracy**: 0.8144

### Gradient Boosting Classifier
- **Training Accuracy**: 0.837
- **Testing Accuracy**: 0.834

### XGBoost Classifier
- **Training Accuracy**: 0.8953
- **Testing Accuracy**: 0.8550

### Voting Classifier
- **Training Accuracy**: 0.9797747795044099
- **Testing Accuracy**: 0.8508299136399828

### AutoML (FLAML)
- **Training Accuracy**: 0.9998293784124318
- **Testing Accuracy**: 0.8648863845164452


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/academic-success-classification.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to see the results.

## Contact

Feel free to reach out if you have any questions or suggestions.

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Reading the data
data = pd.read_csv('path_to_your_data.csv')
test = pd.read_csv('path_to_your_test_data.csv')

# Cleaning and preprocessing
data.drop(columns=['id', 'Nacionality', "Mother's occupation", "Father's occupation", "Displaced", "International", 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (without evaluations)'], inplace=True)
label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])
smote = SMOTE()
X, Y = data.drop(columns='Target'), data['Target']
X_res, Y_res = smote.fit_resample(X, Y)
scaler = MinMaxScaler()
X_res = scaler.fit_transform(X_res)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(X_res, Y_res, train_size=0.7, random_state=42)

# Random Forest Classifier model
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
print("Random Forest - Training Accuracy:", model_rf.score(x_train, y_train))
print("Random Forest - Testing Accuracy:", model_rf.score(x_test, y_test))

# AdaBoost Classifier model
model_ab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=100, min_samples_split=8, min_samples_leaf=4, random_state=42), n_estimators=5, learning_rate=1)
model_ab.fit(x_train, y_train)
print("AdaBoost - Training Accuracy:", model_ab.score(x_train, y_train))
print("AdaBoost - Testing Accuracy:", model_ab.score(x_test, y_test))

# Gradient Boosting Classifier model
model_gb = GradientBoostingClassifier(loss='log_loss', n_estimators=100)
model_gb.fit(x_train, y_train)
print("Gradient Boosting - Training Accuracy:", model_gb.score(x_train, y_train))
print("Gradient Boosting - Testing Accuracy:", model_gb.score(x_test, y_test))

# XGBoost Classifier model
model_xgb = xgb.XGBClassifier()
model_xgb.fit(x_train, y_train)
print("XGBoost - Training Accuracy:", model_xgb.score(x_train, y_train))
print("XGBoost - Testing Accuracy:", model_xgb.score(x_test, y_test))

# Voting Classifier
model_voting = VotingClassifier(estimators=[('rf', model_rf), ('gb', model_gb), ('xgb', model_xgb), ('ab', model_ab)], voting='soft')
model_voting.fit(x_train, y_train)
print("Voting Classifier - Training Accuracy:", model_voting.score(x_train, y_train))
print("Voting Classifier - Testing Accuracy:", model_voting.score(x_test, y_test))
```
