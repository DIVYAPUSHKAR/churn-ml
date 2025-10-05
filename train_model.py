import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Handle TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Select necessary features (including gender)
selected_features = [
    "gender", "tenure", "Contract", "InternetService",
    "MonthlyCharges", "TotalCharges", "PaymentMethod",
    "Partner", "Dependents"
]
X = df[selected_features]
y = df["Churn"]

# Separate categorical + numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, "models/churn_model_short.pkl")
print("✅ Short-feature model with gender saved!")

