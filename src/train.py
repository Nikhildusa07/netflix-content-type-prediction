from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import prepare_features
from src.model_training import train_model, save_model


DATA_PATH = "data/Dataset.csv"


df = load_data(DATA_PATH)
df = clean_data(df)
df = prepare_features(df)

model, vectorizer, X_test, y_test = train_model(df)

save_model(model, vectorizer)

print("Model training completed.")
print("Model saved successfully.")