# https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus
import argparse
import csv
import os
import random
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, CSVWriter


def validate_model(model: LCRRotHopPlusPlus, dataset: EmbeddingsDataset, name='LCR-Rot-hop++'):
    
    # add bootstrap here such that we sample with replacement from the dataset to create a new dataset with same amount of entries
    bootstrap_dataset = []
    n_samples = len(dataset)
    for _ in range(n_samples):
        bootstrap_dataset.append(random.choice(dataset))

    test_loader = DataLoader(bootstrap_dataset, collate_fn=lambda batch: batch)

    print(f"Validating model using embeddings from {dataset}")

    # run validation
    n_classes = 3
    n_correct = [0 for _ in range(n_classes)]
    n_label = [0 for _ in range(n_classes)]
    n_predicted = [0 for _ in range(n_classes)]
    brier_score = 0

    for i, data in enumerate(tqdm(test_loader, unit='obs')):
        torch.set_default_device(dataset.device)

        with torch.inference_mode():
            (left, target, right), label, hops = data[0]

            output: torch.Tensor = model(left, target, right, hops)
            pred = output.argmax(0)
            is_correct: bool = (pred == label).item()

            n_label[label.item()] += 1
            n_predicted[pred.item()] += 1

            if is_correct:
                n_correct[label.item()] += 1

            for j in range(n_classes):
                if (j == label).item():
                    brier_check = 1
                else:
                    brier_check = 0

                p: float = output[j].item()
                brier_score += (p - brier_check) ** 2

        torch.set_default_device('cpu')

    precision = 0
    recall = 0
    for i in range(n_classes):
        if not n_predicted[i] == 0:
            precision += (n_correct[i] / n_predicted[i]) / n_classes
        recall += (n_correct[i] / n_label[i]) / n_classes

    accuracy = (sum(n_correct) / sum(n_label)) * 100

    return accuracy


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    #default is None
    parser.add_argument("--ont-hops", default= 0, type=int, required=False, help="The number of hops in the ontology")
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    #default is None
    parser.add_argument("--gamma", default= None, type=float, required=False,
                        help="The value of gamma for the LCRRotHopPlusPlus model")
    parser.add_argument("--vm", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use the visible matrix")
    parser.add_argument("--sp", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use soft positions")
    parser.add_argument("--usegamma", default = True, type = bool, action = argparse.BooleanOptionalAction,
                        help = "Whether to use gamma")
    parser.add_argument("--ablation", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Run an ablation experiment, this requires all embeddings to exist for a given year.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Path to a state_dict of the LCRRotHopPlusPlus model")
    group.add_argument("--checkpoint", type=str, help="Path to a checkpoint dir from main_hyperparam.py")

    args = parser.parse_args()

    year: int = args.year
    ont_hops: Optional[int] = args.ont_hops
    hops: int = args.hops
    gamma: Optional[float] = args.gamma
    use_vm: bool = args.vm
    use_soft_pos: bool = args.sp
    use_gamma: bool = args.usegamma
    run_ablation: bool = args.ablation

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    model = LCRRotHopPlusPlus(gamma=gamma, hops=hops).to(device)

    if args.model is not None:
        state_dict = torch.load(args.model, map_location=device)
        model.load_state_dict(state_dict)
    elif args.checkpoint is not None:
        state_dict, _ = torch.load(os.path.join(args.checkpoint, "state_dict.pt"), map_location=device)
        model.load_state_dict(state_dict)

   

    model.eval()

    if not run_ablation:
        accuracy_list = []

        for i in range(100): 
            dataset = EmbeddingsDataset(year=year, device=device, phase="Test", ont_hops=ont_hops, use_vm=use_vm,
                                    use_soft_pos=use_soft_pos)
            accuracy = validate_model(model, dataset)

            accuracy_list.append(accuracy)
            print(i)

        print("Average: ", calculate_average(accuracy_list))

        save_accuracy_list(accuracy_list, "./data/bootstrap_results/2015_test_wparam.csv")
        return


def save_accuracy_list(accuracy_list, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for accuracy in accuracy_list:
            writer.writerow([accuracy])

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0 
    else:
        return sum(numbers) / len(numbers)

if __name__ == "__main__":
    main()