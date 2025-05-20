import os
from Utils.data_utils import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from Utils.config import MODEL_DIR
from Utils.model_utils import save_pickle
from sklearn.metrics import r2_score  

# Load and preprocess data
data, label_encoders = load_and_preprocess_data("Data/car_price_dataset.csv")

# Save label encoders
save_pickle(label_encoders, "label_encoders.pkl")

# Define feature columns
feature_columns = data.columns.tolist()
feature_columns.remove("Price")  # Exclude target
save_pickle(feature_columns, "feature_columns.pkl")

# Train-test split
X = data[feature_columns]
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store models and their R² scores
model_results = {}

# Define and train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Save model and score
    model_results[name] = r2
    save_pickle(model, f"{name}.pkl")


save_pickle(model_results, "model_r2_scores.pkl")

print("All models and evaluation scores saved successfully!")