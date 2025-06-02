import argparse
import os
from typing import Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split
from pytorchtools import EarlyStopping
import numpy as np


def stringify_float(value: float):
    return str(value).replace('.', '-')


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    parser.add_argument("--ont-hops", default=0, type=int, 
                       help="The number of hops to use in ontology, including the validation process")
    parser.add_argument("--gamma", default=0.0, type=float,
                       help="The gamma value for ontology hop weighting")
    parser.add_argument("--domain", default="restaurants", type=str,
                       help="The domain to use for the model (restaurants or laptops)")
    parser.add_argument("--val-injection", action="store_true",
                       help="Whether to use knowledge injection or not")
    parser.add_argument("--no-knowledge-injection", action="store_true",
                       help="Whether to use knowledge injection or not")
    args = parser.parse_args()

    year: int = args.year
    lcr_hops: int = args.hops
    ont_hops: int = args.ont_hops
    gamma: float = args.gamma
    domain: str = args.domain
    val_injection: bool = args.val_injection
    no_knowledge_injection: bool = args.no_knowledge_injection
    
    dropout_rate = 0.5
    learning_rate = 0.1
    momentum = 0.95
    weight_decay = 0.001
    n_epochs = 100
    batch_size = 48


    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    # create training anf validation DataLoader
    train_dataset = EmbeddingsDataset(year=year, device=device, phase="Train", domain=domain)
    print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
    train_idx, validation_idx = train_validation_split(train_dataset)

    training_subset = Subset(train_dataset, train_idx)

    validation_subset = Subset(train_dataset, validation_idx)
    print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")

    training_loader = DataLoader(training_subset, batch_size=batch_size, collate_fn=lambda batch: batch)
    validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

    # Train model
    model = LCRRotHopPlusPlus(hops=lcr_hops, dropout_prob=dropout_rate, ont_hops=ont_hops, gamma=gamma, domain=domain).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_accuracy: Optional[float] = None
    best_state_dict: Optional[dict] = None
    epochs_progress = tqdm(range(n_epochs), unit='epoch')

    patience = 15

    models_dir = os.path.join("data", "modelsLayers")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir,
                         f"new_{year}{domain}_ont{ont_hops}_val{val_injection}_noinjection{no_knowledge_injection}_gamma{stringify_float(gamma)}_dropout{stringify_float(dropout_rate)}_acc{stringify_float(best_accuracy)}.pt")
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)

    valid_losses = []

    try:
        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(device)

                if i == 0 and epoch == 0 or no_knowledge_injection:
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

            # ---------Validation loss---------
            epoch_progress = tqdm(validation_loader, unit='obs', leave=False)
            model.eval()

            val_loss = 0.0
            val_steps = 0
            val_n = 0
            val_n_correct = 0
            for i, data in enumerate(epoch_progress):
                torch.set_default_device(device)

                with torch.inference_mode():
                    (sentence, target_index_start, target_index_end), label, hops = data[0]

                    if val_injection:
                        knowledge_layers = range(9,12)
                    else:
                        knowledge_layers = range(-2,-1)
                        
                    output: torch.Tensor = model(sentence, target_index_start, target_index_end, knowledge_layers = knowledge_layers)

                    val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                    val_n += 1

                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_steps += 1

                    valid_losses.append(loss.item())

                    epoch_progress.set_description(
                        f"Test Loss: {val_loss / val_steps:.3f}, Test Acc.: {val_n_correct / val_n:.3f}")

                torch.set_default_device('cpu')


            validation_accuracy = val_n_correct / val_n
            valid_loss = np.average(valid_losses)
            valid_losses = []

            early_stopping(valid_loss, model)

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(f"Best Test Acc.: {validation_accuracy:.3f}")
                best_accuracy = validation_accuracy
                best_state_dict = model.state_dict()

            if early_stopping.early_stop:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Interrupted training procedure, saving best model...")

    if best_state_dict is not None:
        models_dir = os.path.join("data", "modelsLayers")
        os.makedirs(models_dir, exist_ok=True)
        domain_str = f"_domain{domain}"
        model_path = os.path.join(models_dir,
                                  f"new_{year}{domain}_ont{ont_hops}_val{val_injection}_noinjection{no_knowledge_injection}_gamma{stringify_float(gamma)}_dropout{stringify_float(dropout_rate)}_acc{stringify_float(best_accuracy)}.pt")
        with open(model_path, "wb") as f:
            torch.save(best_state_dict, f)
            print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

