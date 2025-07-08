import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    # Example preprocessing steps
    data.fillna(0, inplace=True)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)

def main():
    # Load configuration
    with open(os.path.join('..', 'configs', 'model_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    data = load_data(os.path.join('..', 'data', 'processed', 'training_data.csv'))
    X, y = preprocess_data(data)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Save the trained model
    save_model(model, os.path.join('..', 'models', 'saved', 'trained_model.joblib'))

if __name__ == "__main__":
    main()