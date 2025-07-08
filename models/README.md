# ml-model-repo/models/README.md

# Model Repository

This directory contains the saved versions of the trained AI models. Each model is stored in a separate subdirectory under `saved`, and the naming convention follows the format `model_name_version`, where `model_name` is the name of the model and `version` indicates the version number.

## Saved Models

- **Model Name**: Brief description of the model.
- **Version**: The version number of the model.
- **Date**: The date when the model was trained and saved.
- **Performance Metrics**: Key metrics that indicate the model's performance (e.g., accuracy, precision, recall).

## Usage

To load a saved model for inference, use the following code snippet:

```python
from src.inference.predict import load_model

model = load_model('path/to/saved/model')
```

Ensure that the model path is correctly specified to access the desired version of the model.

## Future Work

- Add more models as they are trained.
- Document the training process and hyperparameters used for each model version.
- Include instructions for fine-tuning and retraining models as needed.