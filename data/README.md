# Data Directory Structure

This directory contains the data used for training and evaluating the AI model. It is organized into two main subdirectories:

## Raw Data
- **Location:** `ml-model-repo/data/raw`
- **Description:** This folder contains the raw, unprocessed data that is used as input for training the model. It is important to keep this data intact for future reference and potential reprocessing.

## Processed Data
- **Location:** `ml-model-repo/data/processed`
- **Description:** This folder contains the processed data that has been cleaned and transformed, making it ready for training. The preprocessing steps applied to the raw data should be documented to ensure reproducibility.

## Usage
To use the data in this repository, follow these guidelines:
1. Place any new raw data files in the `raw` directory.
2. Use the preprocessing scripts located in the `src/preprocessing` directory to transform the raw data into the processed format.
3. Store the processed data in the `processed` directory for training the model.

Make sure to keep the data organized and document any changes made to the data structure or processing methods.