# Time Period Text Classification using the Variational Information Bottleneck

In recent years, text classification has gained significant attention due to its diverse applications across various domains. One promising approach for improving classification accuracy while mitigating overfitting is the Variational Information Bottleneck (VIB) method. By leveraging the VIB method, this thesis aims to enhance the efficiency of neural networks in the context of text classification tasks. Specifically, this thesis focuses on the challenge of classifying the publication time period of books.

To explore this, we divided each book into paragraphs and used word-level embeddings from a pretrained language model as input to a neural network, with each paragraph labelled according to the book’s publication date. We implemented an Encoder-Decoder architecture and tuned various hyper-parameters such as network depth, the $\beta$ parameter in the IB Objective, and $\alpha$ in Rényi Information Measures to optimize our model's prediction accuracy.

The VIB-based model demonstrated competitive accuracy compared to existing solutions, with an test accuracy of 93.89\% for binary classification, and 82.59\% for quaternary classification. The results highlight the efficacy of using the VIB method to learn compressed, yet informative, representations of textual features relevant to publication periods. This work suggests promising directions for integrating information-theoretic constraints into natural language processing pipelines for historical and literary analysis.

Andrew Sutcliffe, Hamish Spence, Kyle Dong, Joshua Geddes, Thomas Tesselaar

The core implementations of this project are in the `vib.py` and `load_gutenberg.py` files, which are commented extensively for future analysis and expansion. The remaining scripts were previous iterations of the projects and scripts designed for assessing various model configurations.

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
