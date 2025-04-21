# LCR-Rot-hop-Kfont++ with Multipule Hops, Domains, Regimes

Source code for expanding LCR-Rot-hop-Kfont++ (a method injecting knowledge from a domain-specific ontology into LCR-Rot-hop++ inspired by work on Kformer for Aspect-Based Sentiment Classification) into multipule hops, domains and more regimes.  

## Setup
- Create environment
   - Create a environment in Python 3.11
   - Install the required packages by running `pip install -r requirements.txt` in the terminal

 
- Download data
  - Download the following files
    - Train and test data of:
      - SemEval-2014 Task 4 Laptop and Restaurant
      - SemEval-2015 Task 12 Restaurant
      - SemEval-2016 Task 5 Restaurant

  - Also add ontology files in these two domains.

  - Add the files to 'data/raw' and rename them to the following: `ABSA{year}_{domain}_{Test\train}.xml`
    for example: `ABSA15_Restaurants_Train.xml`, `ABSA16_Restaurants_Train.xml`, `ABSA14_Laptops_Train.xml`
    For ontology: `ontology_{domain}.owl`
   
- Use OWL2Vec to create vectors of the ontology
   - Clone the OWL2Vec-Star repository  from [Owl2Vec*](https://github.com/KRR-Oxford/OWL2Vec-Star.git)
   - Copy `ontology_embedding.py` file to the OWL2Vec-Star directory
   - Set `default.cfg` file in the OWL2Vec-Star directory (enable reasoner)
   - Initial folder data/embeddings/ontology_laptops/output/ and data/embeddings/ontology_restaurant/output/
   - For this work, the hermit reasoner was used and an embedding size of 100
   - Save the OWL2Vec ontology to 'data/raw'


## Training and validating the model
The following files can be used to obtain the results:

- `main_preprocess.py`: Removes opinions that contain implicit targets and "conflict" polarities and generates the sentences into the correct format for the BERT model to use. To generate all embeddings for a given year, run `python main_preprocess.py --all`. (If need to extend it to more datasets, modify the program to accommodate different dataset structures.)
- `main_hyperparam.py`: This runs the hyperparameter tuning. Change the year, domains, knowledge injection hops and whether inject in validating. For large dataset can choose propotional dataset for searching hyperparameters. 
- `main_train.py`: Train the model for a given set of hyperparameters. Change the year, domains, knowledge injection hops and whether inject in validating. 
- `main_validate.py`: Validate a trained model. Choose whether inject knowledge in validating and the ontology hops in injecting.

This code is base on source code of LCR-Rot-hop-Kfont++ which can be found [here](https://anonymous.4open.science/r/LCR-Rot-hop-Kfont_plus_plus-D6F8/README.md).