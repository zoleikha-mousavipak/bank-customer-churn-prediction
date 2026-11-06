# Bank Customer Churn Prediction

## Live Demo

[Open the Streamlit App](https://bank-customer-churn-prediction-zoli.streamlit.app/)

**Machine Learning Project | Developed by Zoleikha Mousavi Pak**

## Project Overview

This project uses a Random Forest Classifier to predict whether a bank customer is likely to leave the bank.

The goal is to identify customers who may churn based on selected customer attributes.

## Dataset

- Dataset: Bank Customer Churn Records
- Number of Records: 10,000
- Number of Original Features: 18
- Target Variable: `Exited`

## Input Features

The model uses exactly six input features:

- Age
- NumOfProducts
- EstimatedSalary
- Balance
- CreditScore
- Tenure

## Machine Learning

- Algorithm: Random Forest Classifier
- Training/Test Split: 80% / 20%
- Random State: 42
- Class Weight: Balanced

## Model Evaluation

The model was evaluated using the following metrics:

| Metric | Score |
|---|---:|
| Accuracy | 83.70% |
| Precision | 67.83% |
| Recall | 38.24% |
| F1 Score | 48.90% |

## Feature Importance

The feature importance analysis shows the contribution of each input feature to the model's predictions.

| Feature | Importance |
|---|---:|
| Age | 26.52% |
| EstimatedSalary | 18.18% |
| CreditScore | 17.11% |
| Balance | 15.84% |
| NumOfProducts | 13.79% |
| Tenure | 8.55% |

Age is the most important feature in this model.

## Streamlit Application

The application allows users to:

- Enter six customer attributes.
- Predict whether a customer is likely to churn.
- View the churn probability.
- View the model confidence (probability of the predicted class).

## Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- Pickle
- Streamlit
- Git & GitHub

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/zoleikha-mousavipak/bank-customer-churn-prediction.git
   ```

2. Navigate to the project folder:

   ```bash
   cd bank-customer-churn-prediction
   ```

3. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   python3 -m streamlit run app.py
   ```

**Note:** The trained model (`random_forest_model.pkl`) is included in the repository. The original dataset is not included. To retrain the model, place `Customer-Churn-Records.csv` in the project folder and run:

```bash
python3 train_model.py
```

## Project Files

- `app.py` – Streamlit application
- `train_model.py` – Model training and evaluation
- `random_forest_model.pkl` – Saved trained model
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation