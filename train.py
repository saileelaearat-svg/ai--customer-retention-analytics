import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df.fillna(0, inplace=True)

df = pd.get_dummies(df, drop_first=True)

print(df.shape)
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("Churn_Yes", axis=1)
print(X.columns.tolist())
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
import joblib

joblib.dump(model, "churn_model.pkl")
print("Model Saved!")