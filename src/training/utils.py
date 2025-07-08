def load_data(file_path):
    # Function to load data from a given file path
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Function to preprocess the data
    # This can include normalization, handling missing values, etc.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def split_data(data, test_size=0.2):
    # Function to split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data

def save_model(model, file_path):
    # Function to save the trained model to a file
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    # Function to load a trained model from a file
    import joblib
    model = joblib.load(file_path)
    return model