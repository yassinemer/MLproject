import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(csv_path):
    columns = [
        "Brand", "Model", "Year", "Engine_Size", "Fuel_Type", "Transmission",
        "Mileage", "Doors", "Owner_Count", "Price"
    ]
    data = pd.read_csv(csv_path, header=None)
    data.columns = columns

    numeric_cols = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count", "Price"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(inplace=True)

    label_encoders = {}
    for col in ["Brand", "Model", "Fuel_Type", "Transmission"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders
