import joblib

scaler = joblib.load("scaler_X.pkl")

print(scaler.feature_names_in_)

print(FEATURE_COLUMNS)