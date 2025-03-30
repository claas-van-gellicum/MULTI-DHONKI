import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, CSVWriter


def validate_model(model: LCRRotHopPlusPlus, dataset: EmbeddingsDataset, name='LCR-Rot-hop++'):
    test_loader = DataLoader(dataset, collate_fn=lambda batch: batch)

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
            (sentence, target_index_start, target_index_end), label, hops = data[0]

            output: torch.Tensor = model(sentence, target_index_start, target_index_end, knowledge_layers = range(-2,-1))
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

    perf_measures = {
        'Model': name,
        'Correct': sum(n_correct),
        'Accuracy': f'{(sum(n_correct) / sum(n_label)) * 100:.2f}%',
        'Precision': f'{precision * 100:.2f}%',
        'Recall': f'{recall * 100:.2f}%',
        'F1-score': f'{(2 * ((precision * recall) / (precision + recall))) * 100:.2f}%',
        'Brier score': (1 / sum(n_label)) * brier_score
    }

    return perf_measures


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    # default is None
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    # default is None
    parser.add_argument("--ablation", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Run an ablation experiment, this requires all embeddings to exist for a given year.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Path to a state_dict of the LCRRotHopPlusPlus model")
    group.add_argument("--checkpoint", type=str, help="Path to a checkpoint dir from main_hyperparam.py")

    args = parser.parse_args()

    year: int = args.year
    hops: int = args.hops

    run_ablation: bool = args.ablation

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    model = LCRRotHopPlusPlus(hops=hops).to(device)

    if args.model is not None:
        state_dict = torch.load(args.model, map_location=device)
        model.load_state_dict(state_dict)
    elif args.checkpoint is not None:
        state_dict, _ = torch.load(os.path.join(args.checkpoint, "state_dict.pt"), map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    if not run_ablation:
        dataset = EmbeddingsDataset(year=year, device=device, phase="Test")
        result = validate_model(model, dataset)

        print("\nResults:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        return

if __name__ == "__main__":
    main()
