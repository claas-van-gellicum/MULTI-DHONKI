import argparse
import json
import os
import pickle
import random
import numpy as np
from typing import Optional

import torch
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split

class HyperOptManager:
    """A class that performs hyperparameter optimization and stores the best states as checkpoints."""

    def __init__(self, year: int, domain: str, ont_hops: Optional[int] = 0, batch_size: int = 48, lcr_hops: Optional[int] = None, data_proportion: float = 1.0, val_injection: bool = False, no_knowledge_injection: bool = False):
        self.year = year
        self.domain = domain
        self.n_epochs = 20
        self.ont_hops = ont_hops
        self.batch_size = batch_size
        self.lcr_hops = lcr_hops
        self.data_proportion = data_proportion
        self.val_injection = val_injection
        self.no_knowledge_injection = no_knowledge_injection
        self.eval_num = 0
        self.best_loss = None
        self.best_hyperparams = None
        self.best_state_dict = None
        self.trials = Trials()

        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else 'cpu')

        # read checkpoint if exists
        self.__checkpoint_dir = f"data/checkpoints/new_{year}_{domain}_ont{self.ont_hops}_val{self.val_injection}_nok{self.no_knowledge_injection}"
        if self.lcr_hops is not None:
            self.__checkpoint_dir += f"_lcr{self.lcr_hops}"

        if os.path.isdir(self.__checkpoint_dir):
            try:
                self.best_state_dict = torch.load(f"{self.__checkpoint_dir}/state_dict.pt")
                
                with open(f"{self.__checkpoint_dir}/hyperparams.json", "r") as f:
                    self.best_hyperparams = json.load(f)
                
                with open(f"{self.__checkpoint_dir}/model_params.json", "r") as f:
                    model_params = json.load(f)
                    saved_ont_hops = model_params.get("ont_hops", 0)
                    saved_lcr_hops = model_params.get("lcr_hops", None)
                    if saved_ont_hops != self.ont_hops or (saved_lcr_hops is not None and saved_lcr_hops != self.lcr_hops):
                        raise ValueError(f"Checkpoint parameters (ont_hops={saved_ont_hops}, lcr_hops={saved_lcr_hops}) do not match current parameters (ont_hops={self.ont_hops}, lcr_hops={self.lcr_hops})")
                
                with open(f"{self.__checkpoint_dir}/trials.pkl", "rb") as f:
                    self.trials = pickle.load(f)
                    self.eval_num = len(self.trials)
                
                with open(f"{self.__checkpoint_dir}/loss.txt", "r") as f:
                    self.best_loss = float(f.read())

                
                print(f"Resuming from previous checkpoint {self.__checkpoint_dir} with best loss {self.best_loss}")
            except (IOError, ValueError) as e:
                print(f"Error loading checkpoint: {e}")
                raise ValueError(f"Checkpoint {self.__checkpoint_dir} is incomplete or incompatible, please remove this directory")
        
        else:
            print("Starting from scratch")

    def run(self):
        space = [
            hp.choice('learning_rate', [0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
            hp.quniform('dropout_rate', 0.25, 0.75, 0.1),
            hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
            hp.choice('weight_decay', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
        ]
        
        if self.lcr_hops is None:
            space.append(hp.choice('lcr_hops', [3]))
        
        if self.ont_hops > 0:
            space.append(hp.choice('gamma', [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]))

        best = fmin(self.objective, space=space, algo=tpe.suggest, trials=self.trials, show_progressbar=False, max_evals=None) #Change here to limit the number of evaluations

    def objective(self, hyperparams):
        self.eval_num += 1
        
        if self.ont_hops > 0:
            if self.lcr_hops is None:
                learning_rate, dropout_rate, momentum, weight_decay, lcr_hops, gamma = hyperparams
            else:
                learning_rate, dropout_rate, momentum, weight_decay, gamma = hyperparams
                lcr_hops = self.lcr_hops
        else:
            gamma = 0.0
            if self.lcr_hops is None:
                learning_rate, dropout_rate, momentum, weight_decay, lcr_hops = hyperparams
            else:
                learning_rate, dropout_rate, momentum, weight_decay = hyperparams
                lcr_hops = self.lcr_hops
            
        print(f"\n\nEval {self.eval_num} with hyperparams:")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"  - Dropout Rate: {dropout_rate}")
        print(f"  - Momentum: {momentum}")
        print(f"  - Weight Decay: {weight_decay}")
        if self.lcr_hops is None:
            print(f"  - LCR Hops: {lcr_hops}")
        if self.ont_hops > 0:
            print(f"  - Gamma: {gamma}")

        train_dataset = EmbeddingsDataset(year=self.year, domain=self.domain, device=self.device, phase="Train")
        
        dataset_size = len(train_dataset)
        subset_size = int(dataset_size * self.data_proportion)
        indices = list(range(dataset_size))
        random.shuffle(indices) 
        subset_indices = indices[:subset_size]
        
        print(f"Using {train_dataset} with {subset_size} obs ({self.data_proportion * 100:.1f}%) for training and validation")
        
        np.random.seed(42)  
        np.random.shuffle(subset_indices)
        split = int(np.floor(0.2 * len(subset_indices))) 
        train_idx, validation_idx = subset_indices[split:], subset_indices[:split]
        
        training_subset = Subset(train_dataset, train_idx)
        validation_subset = Subset(train_dataset, validation_idx)
        
        print(f"Using {len(training_subset)} obs for training")
        print(f"Using {len(validation_subset)} obs for validation")
        
        training_loader = DataLoader(training_subset, batch_size=self.batch_size, collate_fn=lambda batch: batch)
        validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

        # Train model
        model = LCRRotHopPlusPlus(
            hops=lcr_hops, 
            dropout_prob=dropout_rate,
            domain=self.domain,
            ont_hops=self.ont_hops,
            gamma=gamma
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        best_accuracy: Optional[float] = None
        best_state_dict: Optional[tuple[dict, dict]] = None
        epochs_progress = tqdm(range(self.n_epochs), unit='epoch')

        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(self.device)

                if i == 0 and epoch == 0 or self.no_knowledge_injection:
                    knowledge_layers = range(-2,-1)
                else:
                    knowledge_layers = range(9,12)

                batch_outputs = torch.stack(
                    [model(sentence, target_index_start, target_index_end, knowledge_layers) for (sentence, target_index_start, target_index_end), _, hops in batch], dim=0)
                batch_labels = torch.tensor([label.item() for _, label, _ in batch])

                loss: torch.Tensor = criterion(batch_outputs, batch_labels)

                train_loss += loss.item()
                train_steps += 1
                train_n_correct += (batch_outputs.argmax(1) == batch_labels).type(torch.int).sum().item()
                train_n += len(batch)

                epoch_progress.set_description(
                    f"Train Loss: {train_loss / train_steps:.3f}, Train Acc.: {train_n_correct / train_n:.3f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.set_default_device('cpu')

            # Validation loss
            epoch_progress = tqdm(validation_loader, unit='obs', leave=False)
            model.eval()

            val_loss = 0.0
            val_steps = 0
            val_n = 0
            val_n_correct = 0
            for i, data in enumerate(epoch_progress):
                torch.set_default_device(self.device)

                with torch.no_grad():
                    (sentence, target_index_start, target_index_end), label, hops = data[0]

                    if self.val_injection:
                        knowledge_layers = range(9,12)
                    else:
                        knowledge_layers = range(-2,-1)

                    output: torch.Tensor = model(sentence, target_index_start, target_index_end, knowledge_layers=knowledge_layers)
                    val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                    val_n += 1

                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_steps += 1

                    epoch_progress.set_description(
                        f"Test Loss: {val_loss / val_steps:.3f}, Test Acc.: {val_n_correct / val_n:.3f}")

                torch.set_default_device('cpu')


            validation_accuracy = val_n_correct / val_n

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(f"Best Test Acc.: {validation_accuracy:.3f}")
                best_accuracy = validation_accuracy
                best_state_dict = (model.state_dict(), optimizer.state_dict())

        objective_loss = -best_accuracy
        self.check_best_loss(objective_loss, hyperparams, best_state_dict, lcr_hops)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    def check_best_loss(self, loss: float, hyperparams, state_dict: tuple[dict, dict], lcr_hops: int):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = hyperparams
            self.best_state_dict = state_dict

            os.makedirs(self.__checkpoint_dir, exist_ok=True)

            torch.save(state_dict, f"{self.__checkpoint_dir}/state_dict.pt")
            with open(f"{self.__checkpoint_dir}/hyperparams.json", "w") as f:
                json.dump(hyperparams, f)
            with open(f"{self.__checkpoint_dir}/model_params.json", "w") as f:
                model_params = {
                    "ont_hops": self.ont_hops,
                    "lcr_hops": lcr_hops if self.lcr_hops is None else self.lcr_hops,
                    "batch_size": self.batch_size,
                    "year": self.year,
                    "domain": self.domain
                }
                json.dump(model_params, f)
            with open(f"{self.__checkpoint_dir}/loss.txt", "w") as f:
                f.write(str(self.best_loss))
            print(
                f"Best checkpoint with loss {self.best_loss} and hyperparameters {self.best_hyperparams} saved to {self.__checkpoint_dir}")

        with open(f"{self.__checkpoint_dir}/trials.pkl", "wb") as f:
            pickle.dump(self.trials, f)


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--domain", required=True, type=str, help="The domain of the dataset (e.g., 'laptop', 'restaurant')")
    parser.add_argument("--ont-hops", default=0, type=int, 
                       help="The number of hops to use in ontology, including the validation process")
    parser.add_argument("--batch-size", default=48, type=int, required=False,
                       help="The batch size to use for training")
    parser.add_argument("--lcr-hops", default=None, type=int, required=False,
                       help="The number of hops to use in LCR model (fixed value instead of hyperparameter search)")
    parser.add_argument("--data-proportion", default=1.0, type=float, required=False,
                        help="The proportion of the dataset to use for training and validation (0.0 to 1.0)")
    parser.add_argument("--val-injection", action="store_true",
                        help="Whether to use knowledge injection or not")
    parser.add_argument("--no-knowledge-injection", action="store_true",
                        help="no knowledge injection in training")
    args = parser.parse_args()
    
    year: int = args.year
    domain: str = args.domain
    ont_hops: int = args.ont_hops
    batch_size: int = args.batch_size
    lcr_hops: Optional[int] = args.lcr_hops
    data_proportion: float = args.data_proportion
    val_injection: bool = args.val_injection
    no_knowledge_injection: bool = args.no_knowledge_injection

    opt = HyperOptManager(year=year, domain=domain, ont_hops=ont_hops, batch_size=batch_size, lcr_hops=lcr_hops, data_proportion=data_proportion, val_injection=val_injection, no_knowledge_injection=no_knowledge_injection)
    opt.run()


if __name__ == "__main__":
    main()
