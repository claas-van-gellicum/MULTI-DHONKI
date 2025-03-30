# Aspect-Based Sentiment Classification with the LCR-Rot-hop-ont-plus-plus neural network
This code can be used to train and validate LCR-Rot-hop(-ont)++ models. The model is a neural network used for Aspect-Based Sentiment Classification and
allows for injection knowledge from an Ontology. 

## Before running the code
- Set up environment:
  - Create a conda environment with Python 3.10
  - run `pip install -r requirements.txt` in your terminal 
- Set up data
  - the data and ontologies can be found at `data/raw`  
  - for simplicity the SemEval 2014 laptop dataset is named "ABSA14_Restaurants_...."

## Training and validating process
Note that this process works for the 2015 and 2016 dataset, in the case of the 2014 datasets some adaptations have to be made to the code. These adaptations are explained below. 

- Step 1: Run main_preprocess.py, adapt the year and the amount of ontology hops used as needed.
- Step 2: Run main_hyperparam.py, this code optimizes the hyperparameters and must be run for every specific task before training. Adapt the year and the amount of ontology hops 
          used  as needed (note specify the ontology hops used during training in line 80).
- Step 3: Run main_train.py, use the hyperparameters of the previous step and adapt the year and the amount of ontology hops used as needed.
- Step 4: Run main_validate.py, specify the --model "MODEL_PATH" when running and adapt the year and the amount of ontology hops used as needed. This code will provide the results 
          for  the performance of the model as output. 

## Statistical significance
For assesing the statistical significance the following 2 steps have to be followed

- Step 1: Run main_validate_bootstrap.py, running the code is similar to Step 4, however, 100 bootstraps are performend for which the accuracies are saved (change the file name for
          how it's saved on line 136). The average accuracy is shown after running the code.
- Step 2: Run Wilcoxon_Sign-Ranked_Test.py, change the bootstraps that are compared on line 23 and 24 before running the code.

## Adaption for the 2014 restaurant data
For the SemEval 2014 restaurant dataset the training and validating process is the same as above, however, in Step 1 main_preprocess_restaurant_2014.py is used instead of main_preprocess.py.

## Adaption for the 2014 laptop data
For the SemEval 2014 laptop dataset the training and validating process is the same as above, however, in Step 1 main_preprocess_laptop.py is used instead of main_preprocess.py. Furthermore, `model\ontology.py` has to be replaced by ontology_laptop.py (renaming the files is the easiest approach for this).

## Ontology hop weights as parameters
To use the LCR-Rot-hop(-ont)++ model where the onotlogy hop weights are parameters of the model, `model\lcr_rot_hop_plus_plus.py` has to be replaced by lcr_rot_hop_plus_plus_hopweight_param.py (renaming the files is the easiest approach for this).

## References
Code is used from:
- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus 
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus/tree/c8d18b8b847a0872bd66d496e00d7586fdffb3db.
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
