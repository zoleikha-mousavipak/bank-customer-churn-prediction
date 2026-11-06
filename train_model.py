
# 1. Import Libraries
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 2. Load Dataset
df = pd.read_csv("Customer-Churn-Records.csv")

# 3. Select Six Input Features
features = [
    "Age",
    "NumOfProducts",
    "EstimatedSalary",
    "Balance",
    "CreditScore",
    "Tenure"
]

X = df[features]
y = df["Exited"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)

print("Model Evaluation")
print("----------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# 7. Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance")
print(importance)

# 8. Save Trained Model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
