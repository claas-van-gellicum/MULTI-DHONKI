# https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus
import argparse
import os
from typing import Optional
import xml.etree.ElementTree as ElementTree

import torch
from rdflib import Graph
from tqdm import tqdm

from model import EmbeddingsLayer
from utils import download_from_url, EmbeddingsDataset


def clean_data(year: int, phase: str, domain: str):
    """Clean a SemEval dataset by removing opinions with implicit targets. This function returns the cleaned dataset."""
    #filename = f"ABSA{year % 2000}_Restaurants_{phase}.xml"
    filename = f"ABSA{year % 2000}_{domain}_{phase}.xml"

    input_path = f"data/raw/{filename}"
    output_path = f"data/processed/{filename}"

    if os.path.isfile(output_path):
        print(f"Found cleaned file at {output_path}")
        return ElementTree.parse(output_path)

    tree = ElementTree.parse(input_path)

    if year == 2015 and domain == "Laptop":
        raise ValueError(f"Unknown domain and year combination for {domain} dataset and year {year}. Laptop exist only for 2014, Restaurant exists for 2014, 2015, 2016.")
    
    elif (year == 2016 and domain == "Laptop"):
        raise ValueError(f"Unknown domain and year combination for {domain} dataset and year {year}. Laptop exist only for 2014, Restaurant exists for 2014, 2015, 2016.")
    
    elif (year == 2014 and domain == "Restaurant"):
        # remove implicit targets
        n_conflict_removed = 0
        for aspectterms in tree.findall(".//aspectTerms"):
            for aspectterm in aspectterms.findall('./aspectTerm[@polarity="conflict"]'):
                aspectterms.remove(aspectterm)
                n_conflict_removed += 1

        # calculate descriptive statistics for remaining aspect terms
        n = 0
        n_positive = 0
        n_negative = 0
        n_neutral = 0
        for aspect_term in tree.findall(".//aspectTerm"):
            n += 1

            if aspect_term.attrib['polarity'] == "positive":
                n_positive += 1
            elif aspect_term.attrib['polarity'] == "negative":
                n_negative += 1
            elif aspect_term.attrib['polarity'] == "neutral":
                n_neutral += 1

        if n == 0:
            print(f"\n{filename} does not contain any aspect terms")
        else:
            print(f"\n{filename}")
            print(f"  Removed {n_conflict_removed} with conflict polarity")
            print(f"  Total number of aspect terms: {n}")
            print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
            print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
            print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path)
        print(f"Stored cleaned dataset in {output_path}")

    elif (year == 2015 or year == 2016) and domain == "Restaurant":
        # remove implicit targets
        n_null_removed = 0
        for opinions in tree.findall(".//Opinions"):
            for opinion in opinions.findall('./Opinion[@target="NULL"]'):
                opinions.remove(opinion)
                n_null_removed += 1

        # calculate descriptive statistics for remaining opinions
        n = 0
        n_positive = 0
        n_negative = 0
        n_neutral = 0
        for opinion in tree.findall(".//Opinion"):
            n += 1

            if opinion.attrib['polarity'] == "positive":
                n_positive += 1
            elif opinion.attrib['polarity'] == "negative":
                n_negative += 1
            elif opinion.attrib['polarity'] == "neutral":
                n_neutral += 1

        if n == 0:
            print(f"\n{filename} does not contain any opinions")
        else:
            print(f"\n{filename}")
            print(f"  Removed {n_null_removed} opinions with target NULL")
            print(f"  Total number of opinions remaining: {n}")
            print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
            print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
            print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path)
        print(f"Stored cleaned dataset in {output_path}")

    elif year == 2014 and domain == "Laptop":
        # remove implicit targets
        n_null_removed = 0
        for aspectterms in tree.findall(".//aspectTerms"):
            for aspectterm in aspectterms.findall('./aspectTerm[@polarity="conflict"]'):
                aspectterms.remove(aspectterm)
                n_null_removed += 1

        # calculate descriptive statistics for remaining aspect terms
        n = 0
        n_positive = 0
        n_negative = 0
        n_neutral = 0
        for aspect_term in tree.findall(".//aspectTerm"):
            n += 1

            if aspect_term.attrib['polarity'] == "positive":
                n_positive += 1
            elif aspect_term.attrib['polarity'] == "negative":
                n_negative += 1
            elif aspect_term.attrib['polarity'] == "neutral":
                n_neutral += 1

        if n == 0:
            print(f"\n{filename} does not contain any aspect terms")
        else:
            print(f"\n{filename}")
            print(f"  Removed {n_null_removed} with conlfict polarity")
            print(f"  Total number of aspect terms: {n}")
            print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
            print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
            print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path)
        print(f"Stored cleaned dataset in {output_path}")

    return tree
    
    
def generate_embeddings(embeddings_layer: EmbeddingsLayer, data: ElementTree, embeddings_dir: str, year: int, domain: str):
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"\nGenerating embeddings into {embeddings_dir}")

    labels = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
    }

    
    if year == 2014 and domain == "Laptop":
        os.makedirs(embeddings_dir, exist_ok=True)
        print(f"\nGenerating embeddings into {embeddings_dir}")

        labels = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }

        with torch.inference_mode():
            i = 0
            for node in tqdm(data.findall('.//sentence'), unit='sentence'):
                sentence = node.find('./text').text

                for aspect_term in node.findall('.//aspectTerm'):
                    target_from = int(aspect_term.attrib['from'])
                    target_to = int(aspect_term.attrib['to'])
                    polarity = aspect_term.attrib['polarity']

                    if polarity not in labels:
                        raise ValueError(f"Unknown polarity \"{polarity}\" found at sentence \"{sentence}\"")

                    label = labels.get(polarity)
                    embeddings, target_pos, hops = embeddings_layer.forward(sentence, target_from, target_to)
                    data = {'label': label, 'embeddings': embeddings, 'target_pos': target_pos, 'hops': hops}
                    torch.save(data, f"{embeddings_dir}/{i}.pt")
                    i += 1
            print(f"Generated embeddings for {i} aspect terms")

    elif year == 2014 and domain == "Restaurant":

        with torch.inference_mode():
            i = 0
            for node in tqdm(data.findall('.//sentence'), unit='sentence'):
                sentence = node.find('./text').text

                for aspect_term in node.findall('.//aspectTerm'):
                    target_from = int(aspect_term.attrib['from'])
                    target_to = int(aspect_term.attrib['to'])
                    polarity = aspect_term.attrib['polarity']

                    if polarity not in labels:
                        raise ValueError(f"Unknown polarity \"{polarity}\" found at sentence \"{sentence}\"")

                    label = labels.get(polarity)
                    embeddings, target_pos, hops = embeddings_layer.forward(sentence, target_from, target_to)
                    data = {'label': label, 'embeddings': embeddings, 'target_pos': target_pos, 'hops': hops}
                    torch.save(data, f"{embeddings_dir}/{i}.pt")
                    i += 1

            print(f"Generated embeddings for {i} aspect terms")
    elif (year == 2015 or 2016) and domain == "Restaurant":

        with torch.inference_mode():
            i = 0
            for node in tqdm(data.findall('.//sentence'), unit='sentence'):
                sentence = node.find('./text').text

                for opinion in node.findall('.//Opinion'):
                    target_from = int(opinion.attrib['from'])
                    target_to = int(opinion.attrib['to'])
                    polarity = opinion.attrib['polarity']

                    if polarity not in labels:
                        raise ValueError(f"Unknown polarity \"{polarity}\" found at sentence \"{sentence}\"")

                    label = labels.get(polarity)
                    embeddings, target_pos, hops = embeddings_layer.forward(sentence, target_from, target_to)
                    data = {'label': label, 'embeddings': embeddings, 'target_pos': target_pos, 'hops': hops}
                    torch.save(data, f"{embeddings_dir}/{i}.pt")
                    i += 1

            print(f"Generated embeddings for {i} opinions")


def load_ontology(domain: str):
    if domain == "Restaurant":
        #path = download_from_url(
            #url="https://raw.githubusercontent.com/KSchouten/Heracles/master/src/main/resources/externalData/ontology.owl",
            #path="./data/raw/ontology.owl-Extended.owl")
            path ="./data/raw/restaurant.owl"
    elif domain =="Laptop":
        path = "./data/raw/laptop.owl" 

    return Graph().parse(path)


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--domain", default="Restaurant", help="The domain of the dataset (Restaurant or Laptop)")
    parser.add_argument("--ont-hops", default=None, type=int, required=False,
                        help="The number of hops in the ontology to use")
    parser.add_argument("--vm", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use the visible matrix")
    parser.add_argument("--sp", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="Whether to use soft positions")
    parser.add_argument("--all", default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help="Generate all embeddings for a given year")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    domain: str = args.domain
    ont_hops: Optional[int] = args.ont_hops
    use_vm: bool = args.vm
    use_soft_pos: bool = args.sp
    generate_all: bool = args.all


    if ont_hops is None and (use_vm is False or use_soft_pos is False):
        raise ValueError("The visible matrix and soft positions have no effect without hops in the ontology")

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_default_device(device)

    # generate embeddings only for selected options
    if not generate_all:
        data = clean_data(year, phase, domain)

        # load ontology
        ontology: Graph | None = None
        if ont_hops is not None and ont_hops >= 0:
            print(f"Loading ontology to include {ont_hops} hops")
            ontology = load_ontology(domain)

        embeddings_layer = EmbeddingsLayer(hops=ont_hops, ontology=ontology, use_vm=use_vm, use_soft_pos=use_soft_pos,
                                           device=device)
        embeddings_dir = EmbeddingsDataset(year=year, device=device, phase=phase, domain = domain, ont_hops=ont_hops, empty_ok=True,
                                           use_vm=use_vm, use_soft_pos=use_soft_pos).dir
        generate_embeddings(embeddings_layer, data, embeddings_dir, year, domain)
        return

    print(f"\nGenerating all embeddings for year {year}")
    ontology = load_ontology(domain)

    for phase in ['Train', 'Test']:
        data = clean_data(year, phase, domain)
        embeddings_layer = EmbeddingsLayer(hops=None, ontology=ontology, device=device)
        embeddings_dir = EmbeddingsDataset(year=year, device=device, phase=phase, domain=domain, ont_hops=None, empty_ok=True).dir
        generate_embeddings(embeddings_layer, data, embeddings_dir, year, domain)

        
        # inject knowledge for test datasets
        for ont_hops in range(3):
            for use_vm in [True, False]:
                for use_soft_pos in [True, False]:
                    embeddings_layer = EmbeddingsLayer(hops=ont_hops, ontology=ontology, use_vm=use_vm,
                                                       use_soft_pos=use_soft_pos, device=device)
                    embeddings_dir = EmbeddingsDataset(year=year, device=device, phase=phase, domain = domain, ont_hops=ont_hops,
                                                       empty_ok=True, use_vm=use_vm, use_soft_pos=use_soft_pos).dir
                    generate_embeddings(embeddings_layer, data, embeddings_dir, year, domain)


if __name__ == "__main__":
    main()