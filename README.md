
# Bank Customer Churn Prediction

**Machine Learning Project | Developed by Zoleikha Mousavi Pak**

## Project Overview
This project uses a Random Forest Classifier to predict whether a bank customer is likely to leave the bank.

## Dataset
- 10,000 customer records
- 6 selected input features
- Target: `Exited` (0 = No Churn, 1 = Churn)

## Input Features
- Age
- NumOfProducts
- EstimatedSalary
- Balance
- CreditScore
- Tenure

## Machine Learning
- Algorithm: Random Forest Classifier
- Train/Test Split: 80% / 20%
- Class Weight: Balanced

## Model Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 83.70% |
| Precision | 67.83% |
| Recall | 38.24% |
| F1 Score | 48.90% |

## Feature Importance
Age was the most important feature in the trained model.

## Streamlit Application
The application allows users to enter six customer attributes and view:
- Churn Prediction
- Churn Probability
- Model Confidence

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

## Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the application:

```bash
python3 -m streamlit run app.py
```

## Project Files
- `app.py` – Streamlit application
- `random_forest_model.pkl` – Trained model
- `requirements.txt` – Project dependencies
- `train_model.py` – Model training and evaluation
- `README.md` – Project documentation
