# Employee-salary-prediction
A ML mini project predicts whether an employee earns >$50K using the UCI Adult dataset

## Overview
Using demographic and employment data from the 1994 US Census, this project trains a binary classifier to predict income bracket. It covers the full ML workflow — from raw data cleaning to model evaluation.

## Dataset
UCI Adult Census Income — available on Kaggle
Features include age, education, occupation, marital status, hours worked per week, and capital gains/losses.
~48,000 records after cleaning
Binary target: ≤$50K (0) or >$50K (1)

## Approach
### Data Cleaning
The raw dataset uses `?` for missing values. These were replaced with NaN and dropped.
Feature Engineering
Two new features were created to improve model signal:
`capital_net` — net capital (gain minus loss) combined into one feature
`work_hours_level` — hours per week bucketed into Low / Medium / High
Education was also ordinally ordered from Preschool through Doctorate.

## Model
A Logistic Regression model was built inside an sklearn Pipeline with a `ColumnTransformer` for preprocessing — one-hot encoding categorical columns and passing numerical ones through directly.

## Results
The model achieved strong baseline performance on the test set, with Logistic Regression serving as an interpretable and efficient classifier for this tabular dataset.

## Project Structure
```
Employee-Salary-Prediction/
│
├── employee_salary_pred.ipynb   
├── requirements.txt             
└── README.md
```

Background
This project was completed as a mini project during an ML internship, applying classical machine learning techniques to a real-world tabular classification problem.
