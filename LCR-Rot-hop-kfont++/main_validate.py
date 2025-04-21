import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.lcr_rot_hop_plus_plus import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, CSVWriter


def validate_model(model: LCRRotHopPlusPlus, dataset: EmbeddingsDataset, name='LCR-Rot-hop++', no_knowledge_injection=False):
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

        try:
            with torch.inference_mode():
                (sentence, target_index_start, target_index_end), label, hops = data[0]

                if no_knowledge_injection:
                    knowledge_layers_to_use = range(-2, -1)
                else:
                    knowledge_layers_to_use = range(9, 12) 
                
                output: torch.Tensor = model(sentence, target_index_start, target_index_end, knowledge_layers=knowledge_layers_to_use)
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
        
        except RuntimeError as e:
            print(f"\nSkipping sample {i} due to error: {e}")
            torch.set_default_device('cpu')
            continue 

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
    parser.add_argument("--domain", default=None, type=str,
                        help="The domain of the dataset (restaurants or laptops)")
    # default is None
    parser.add_argument("--ablation", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Run an ablation experiment, this requires all embeddings to exist for a given year.")
    parser.add_argument("--no-knowledge-injection", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Run an ablation experiment, this requires all embeddings to exist for a given year.")
    parser.add_argument("--gamma", default=0.0, type=float,
                        help="The gamma value for the knowledge injection")
    parser.add_argument("--ont-hops", default=0, type=int,
                        help="The number of hops to use in injection ontology")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Path to a state_dict of the LCRRotHopPlusPlus model")
    group.add_argument("--checkpoint", type=str, help="Path to a checkpoint dir from main_hyperparam.py")

    args = parser.parse_args()

    year: int = args.year
    hops: int = args.hops
    domain: str = args.domain
    no_knowledge_injection: bool = args.no_knowledge_injection
    gamma: float = args.gamma
    ont_hops: int = args.ont_hops
    run_ablation: bool = args.ablation

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    model = LCRRotHopPlusPlus(hops=hops, domain=domain, gamma=gamma, ont_hops=ont_hops).to(device)

    if args.model is not None:
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    elif args.checkpoint is not None:
        if args.checkpoint.endswith("state_dict.pt"):
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(args.checkpoint, "state_dict.pt")
        state_dict, _ = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.eval()

    if not run_ablation:
        dataset = EmbeddingsDataset(year=year, domain=domain, device=device, phase="Test")
        result = validate_model(model, dataset, no_knowledge_injection=no_knowledge_injection)

        print("\nResults:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        return

if __name__ == "__main__":
    main()
