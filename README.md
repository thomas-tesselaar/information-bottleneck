# Time Period Text Classification using the Variational Information Bottleneck

Write abstract here

Andrew Sutcliffe, Hamish Spence, Kyle Dong, Joshua Geddes, Thomas Tesselaar

## Setup Instructions

### 1. Create and Activate Virtual Environment
#### Windows (Command Prompt):
```sh
python -m venv venv
venv\Scripts\activate
```
#### macOS/Linux (Terminal):
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run VIB Script
```sh
python vib.py
```

### 4. Deactivate Virtual Environment (Optional)
```sh
deactivate
```

## File Descriptions

| File Name                 | Description                                                                                  |
|---------------------------|----------------------------------------------------------------------------------------------|
| `adam_vib.py`             | Implementation of VIB from a previous student research project focused on spam classification.           |
| `alpha_tuning.py`         | Script for tuning the alpha parameter in the VIB model.                                      |
| `alpha_tuning_results.csv`| Stores results from the alpha tuning experiments.                                            |
| `beta_accuracy.csv`       | Contains accuracy metrics corresponding to different beta values in the VIB model.           |
| `gutenberg_vib.py`        | Script for training VIB model on various beta values.                               |
| `load_gutenberg.py`       | Utility functions for loading and preprocessing the Gutenberg dataset.                       |
| `model_results.csv`       | Records performance metrics of various model configurations.                                 |
| `model_results2.csv`      | Additional results from alternate model config.                                                   |
| `requirements.txt`        | Lists the Python dependencies required to run the project.                                   |
| `spam_data.csv`           | Dataset containing spam detection data used in the project.                                  |
| `spam_detection_vib.py`   | Implements the VIB model for spam detection tasks.                                           |
| `transformer.py`          | Transformer experiments with VIB on Gutenberg dataset                     |
| `vib.py`                  | Core implementation of the Variational Information Bottleneck model.                         |
