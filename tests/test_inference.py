import pytest
from src.inference.predict import load_model, make_prediction

def test_load_model():
    model = load_model('models/saved/model.h5')
    assert model is not None, "Model should be loaded successfully"

def test_make_prediction():
    model = load_model('models/saved/model.h5')
    sample_input = [[0.5, 0.2, 0.1]]  # Example input
    prediction = make_prediction(model, sample_input)
    assert prediction is not None, "Prediction should not be None"
    assert isinstance(prediction, list), "Prediction should be a list"