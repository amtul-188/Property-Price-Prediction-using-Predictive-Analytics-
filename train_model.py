import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

# Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# Drop ID column
df = df.drop(columns=["Id"])

# Split features & target
X = df.drop(columns=["Price"])   # ⚠️ Make sure column name is EXACT
y = df["Price"]

# Define columns
num_features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
cat_features = ["Location", "Condition", "Garage"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)

# Train model
model = SVR(kernel="rbf")
model.fit(X_train_processed, y_train)

# Save model & preprocessor
joblib.dump(model, "best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model and preprocessor saved successfully")