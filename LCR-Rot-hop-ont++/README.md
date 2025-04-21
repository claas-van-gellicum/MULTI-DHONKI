# Aspect-Based Sentiment Classification with the LCR-Rot-hop-ont-plus-plus Neural Network

This repository contains code to train, validate, and evaluate LCR-Rot-hop-ont++ models for Aspect-Based Sentiment Classification (ABSC), with support for both restaurant and laptop domains. The model allows for knowledge injection from a domain ontology, and includes improvements for flexible domain configuration and end-to-end automation.

## Environment Setup

1. Create a conda environment with Python 3.10  
2. Install dependencies with: `pip install -r requirements.txt`


## Dataset & Ontology Setup

All datasets and ontologies are located in `data/raw/`. You do not need to download anything manually.

## Switching Between Domains

This project supports both restaurant and laptop domains. To switch domains, the correct ontology loader must be configured. This is done via `ontology.py`.

By default, `model/ontology.py` is configured for the restaurant domain. If you are working with the laptop domain, replace the contents of `model/ontology.py` with the contents of `model/ontology_laptop.py`.

## Training and Evaluation Pipeline

The training and validation process consists of four steps. Each step requires the configuration of parameters such as dataset year, ontology hops, and domain.

### Step 1: Preprocessing

Run `main_preprocess.py`. This step cleans the data and generates the embeddings. Specify the year, domain, phase in which to inject knowledge and the amount of ontology hops.

### Step 2: Hyperparameter Optimization

Run `main_hyperparam.py`. This script performs hyperparameter tuning for a specific dataset and ontology configuration. Specify the year, domain, and the number of hops to use in the training and validation phase. 

### Step 3: Training

Run `main_train.py`. This trains the model using the optimized hyperparameters obtained in step 2. Again, specify the year, domain, and the number of hops to use in the training and validation phase.

### Step 4: Validation

Run `main_validate.py`. This will evaluate the trained model on the test set and output accuracy, precision, recall, F1-score, and Brier score. Specify the path to the trained model. Also specify the year, domain and amount of ontology hops of the embeddings you want to test the trained model on.


### Alternative: Running All Steps Automatically

To automatically execute the entire pipeline for training and evaluating all model configurations, you can run `run_all.py`. This script sequentially performs preprocessing, hyperparameter tuning and training for all predefined combinations of domain, year, and ontology hops. This allows you to reproduce all results with a single command, except for obtaining validation results. Refer to `main_validate.py` as some parameters (such as gamma) need to be set manually.

## References

Code is used from:

- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus  
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus  
- https://github.com/StijnCoremans/LCR-Rot-hop-ont-plus-plus