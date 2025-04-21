# Aspect-Based Sentiment Classification with the LCR-Rot-hop-ont-plus-plus Neural Network

This repository contains code to train, validate, and evaluate LCR-Rot-hop(-ont)++ models for Aspect-Based Sentiment Classification (ABSC), with support for both restaurant and laptop domains. The model allows for knowledge injection from a domain ontology, and includes improvements for flexible domain configuration and end-to-end automation.

## Environment Setup

1. Create a conda environment with Python 3.10  
2. Install dependencies with:
   pip install -r requirements.txt


## Dataset & Ontology Setup

All datasets and ontologies are located in `data/raw/`. You do not need to download anything manually.

## Switching Between Domains

This project supports both restaurant and laptop domains. To switch domains, the correct ontology loader must be configured. This is done via `ontology.py`.

By default, `model/ontology.py` is configured for the restaurant domain. If you are working with the laptop domain, replace the contents of `model/ontology.py` with the contents of `model/ontology_laptop.py`.

## Training and Evaluation Pipeline

The training and validation process consists of four steps. Each step requires the configuration of parameters such as dataset year, ontology hops, and domain.

### Step 1: Preprocessing

Run `main_preprocess.py`. In addition to the year and ontology hops, you also specify the domain you wish to use (restaurant or laptop). This step will generate the necessary embeddings.

### Step 2: Hyperparameter Optimization

Run `main_hyperparam.py`. This script performs hyperparameter tuning for a specific dataset and ontology configuration. As in the preprocessing step, the domain must be specified.

### Step 3: Training

Run `main_train.py` using the optimized hyperparameters obtained in step 2. Again, specify the correct dataset year, ontology hops, and domain.

### Step 4: Validation

Run `main_validate.py`. This will evaluate the trained model on the test set and output accuracy, precision, recall, F1-score, and Brier score.

## Statistical Significance Testing

To assess statistical significance, two additional steps can be followed.

### Step 1: Bootstrap Evaluation

Run `main_validate_bootstrap.py`. This script evaluates the model on 100 bootstrap samples and saves the accuracy values for each run. You can modify the filename on line 136 of the script to control where the results are saved.

### Step 2: Wilcoxon Signed-Rank Test

Run `Wilcoxon_Sign-Ranked_Test.py`. Before running, set the paths to the bootstrap result files you want to compare on lines 23 and 24.

## End-to-End Automation

To automate the entire training and evaluation process, a script named `run_all.py` is included. This script executes all necessary steps sequentially for multiple domain and ontology hop configurations. Running this script will perform preprocessing, hyperparameter tuning, model training, and validation in one go.

## Notes

- No manual renaming of files is needed to switch domains.
- Ontology selection is handled via `ontology.py`.
- The pipeline supports multiple domains in a unified and automated structure.

## References

This work is inspired and adapted from:

- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus  
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus  
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)  
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py


