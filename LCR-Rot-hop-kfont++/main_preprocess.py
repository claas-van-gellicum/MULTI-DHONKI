import argparse
import os
from typing import Optional, Tuple
import xml.etree.ElementTree as ElementTree

import torch
from rdflib import Graph
from tqdm import tqdm

from model import EmbeddingsLayer
from utils import download_from_url, EmbeddingsDataset


def _clean_data_2014(tree: ElementTree.ElementTree, filename: str) -> Tuple[ElementTree.ElementTree, int, int]:

    n_null_removed = 0
    n_conflict_removed = 0

    for sentence in tree.findall('.//sentence'):
        # aspectTerms
        aspect_terms = sentence.find('./aspectTerms')
        if aspect_terms is not None:
            null_aspects = aspect_terms.findall('./aspectTerm[@term="NULL"]')
            for aspect in null_aspects:
                aspect_terms.remove(aspect)
                n_null_removed += 1

            conflict_aspects = aspect_terms.findall('./aspectTerm[@polarity="conflict"]')
            for aspect in conflict_aspects:
                aspect_terms.remove(aspect)
                n_conflict_removed += 1

            if len(aspect_terms) == 0 or len(aspect_terms.findall('*')) == 0:
                sentence.remove(aspect_terms)

    return tree, n_null_removed, n_conflict_removed


def _clean_data_2015(tree: ElementTree.ElementTree, filename: str) -> Tuple[ElementTree.ElementTree, int]:

    n_null_removed = 0
    
    for opinions in tree.findall('.//Opinions'):
        for opinion in opinions.findall('./Opinion[@target="NULL"]'):
            opinions.remove(opinion)
            n_null_removed += 1
    return tree, n_null_removed


def _clean_data_2016(tree: ElementTree.ElementTree, filename: str) -> Tuple[ElementTree.ElementTree, int, int]:
    
    conflict_removed = 0
    null_removed = 0
    
    for review in tree.findall('.//Review'):
        sentences = review.find('./sentences')
        if sentences is None:
            continue
        for sentence in sentences.findall('./sentence'):
            opinions_elem = sentence.find('./Opinions')
            if opinions_elem is None:
                continue
            # conflict
            conflict_opinions = opinions_elem.findall('./Opinion[@polarity="conflict"]')
            for op in conflict_opinions:
                opinions_elem.remove(op)
                conflict_removed += 1
            # NULL
            null_ops = opinions_elem.findall('./Opinion[@target="NULL"]')
            for op in null_ops:
                opinions_elem.remove(op)
                null_removed += 1
    return tree, conflict_removed, null_removed


def clean_data(year: int, phase: str, domain: str, force: bool = False) -> ElementTree.ElementTree:

    domain_lower = domain.lower()
    domain_title = domain_lower.title()  
    filename = f"ABSA{year % 2000}_{domain_title}_{phase}.xml"
    input_path = f"data/raw/{filename}"
    output_path = f"data/processed/ABSA{year % 2000}_{domain_lower}_{phase}.xml"

    if force and os.path.isfile(output_path):
        os.remove(output_path)

    tree = ElementTree.parse(input_path)

    if year == 2014:
        tree, n_null_removed, n_conflict_removed = _clean_data_2014(tree, filename)
    elif year == 2015:
        tree, n_null_removed = _clean_data_2015(tree, filename)
        n_conflict_removed = 0
    elif year == 2016:
        tree, n_conflict_removed, n_null_removed = _clean_data_2016(tree, filename)


    n_positive = n_negative = n_neutral = n_conflict = 0
    n_aspectTerm = n_aspectCategory = 0

    if year == 2014:
        
        aspect_terms = tree.findall('.//aspectTerm')
        n_aspectTerm = len(aspect_terms)
        
        for aspect in aspect_terms:
            polarity = aspect.attrib.get('polarity')
            if polarity == 'positive':
                n_positive += 1
            elif polarity == 'negative':
                n_negative += 1
            elif polarity == 'neutral':
                n_neutral += 1
            elif polarity == 'conflict':
                n_conflict += 1
        
        aspect_categories = tree.findall('.//aspectCategory')
        n_aspectCategory = len(aspect_categories)
        
    else:
        
        opinions = tree.findall('.//Opinion')
        n_total = len(opinions)
        
        for op in opinions:
            polarity = op.attrib.get('polarity')
            if polarity == 'positive':
                n_positive += 1
            elif polarity == 'negative':
                n_negative += 1
            elif polarity == 'neutral':
                n_neutral += 1
            elif polarity == 'conflict':
                n_conflict += 1

    print(f"\n{filename}")
    print(f"  Removed {n_null_removed} opinions with target NULL") 
    print(f"  Removed {n_conflict_removed} opinions with conflict polarity")
    print(f"  Total number of opinions remaining: {n_total}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)
    print(f"Stored cleaned dataset in {output_path}")

    return tree


def _generate_embeddings_2014(data: ElementTree.ElementTree, embeddings_dir: str, force: bool = False):
    
    os.makedirs(embeddings_dir, exist_ok=True)
    
    if force:
        for f in [f for f in os.listdir(embeddings_dir) if f.endswith('.pt')]:
            try:
                os.remove(os.path.join(embeddings_dir, f))
            except Exception:
                pass
    print(f"\nGenerating embeddings into {embeddings_dir}")

    labels = {'negative': 0, 'neutral': 1, 'positive': 2}

    with torch.inference_mode():
        i = 0
        for node in tqdm(data.findall('.//sentence'), unit='sentence'):
            
            sentence_text = node.find('./text').text if node.find('./text') is not None else None
            
            if not sentence_text:
                continue
            aspect_terms = node.findall('.//aspectTerm')
            
            for aspect in aspect_terms:
                polarity = aspect.attrib.get('polarity')
                if polarity not in labels:
                    continue
                target_from = int(aspect.attrib['from'])
                target_to = int(aspect.attrib['to'])
                label = labels[polarity]
                opinion_data = {
                    'sentence': sentence_text,
                    'target_from': target_from,
                    'target_to': target_to,
                    'label': label,
                }
                torch.save(opinion_data, f"{embeddings_dir}/{i}.pt")
                i += 1
        
        print(f"Saved embeddings for {i} opinions")


def _generate_embeddings_opinion(data: ElementTree.ElementTree, embeddings_dir: str, force: bool = False):
    
    os.makedirs(embeddings_dir, exist_ok=True)
    
    if force:
        for f in [f for f in os.listdir(embeddings_dir) if f.endswith('.pt')]:
            try:
                os.remove(os.path.join(embeddings_dir, f))
            except Exception:
                pass
    
    labels = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    with torch.inference_mode():
        i = 0
        for node in tqdm(data.findall('.//sentence'), unit='sentence'):
            
            sentence_text = node.find('./text').text if node.find('./text') is not None else None
            
            if sentence_text is None:
                continue
            
            for opinion in node.findall('.//Opinion'):
                polarity = opinion.attrib.get('polarity')
                if polarity not in labels:
                    continue
                
                target_from = int(opinion.attrib.get('from', 0))
                target_to = int(opinion.attrib.get('to', len(sentence_text)))
                
                label = labels[polarity]
                opinion_data = {
                    'sentence': sentence_text,
                    'target_from': target_from,
                    'target_to': target_to,
                    'label': label,
                }
                torch.save(opinion_data, f"{embeddings_dir}/{i}.pt")
                i += 1
        
        print(f"Saved embeddings for {i} opinions")


def generate_embeddings(data: ElementTree.ElementTree, embeddings_dir: str, year: int, force: bool = False):

    if year == 2014:
        _generate_embeddings_2014(data, embeddings_dir, force=force)
    else:
        _generate_embeddings_opinion(data, embeddings_dir, force=force)


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2014, 2015 or 2016)")
    parser.add_argument("--phase", default="Test", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--domain", default="laptops", help="The domain of the dataset (laptops or restaurants)")
    parser.add_argument("--all", default=False, action=argparse.BooleanOptionalAction, required=False,
                        help="Generate all embeddings for all years, domains and phases")
    parser.add_argument("--force", default=False, action=argparse.BooleanOptionalAction, required=False,
                        help="Force regeneration of processed files even if they already exist")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    domain: str = args.domain
    generate_all: bool = args.all
    force: bool = args.force

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_default_device(device)

    if not generate_all:
        data = clean_data(year, phase, domain, force=force)
        embeddings_dir = EmbeddingsDataset(year=year, device=device, phase=phase, domain=domain, empty_ok=True).dir
        generate_embeddings(data, embeddings_dir, year, force=force)
        return
    else:
        print("\nGenerating embeddings for all available datasets")

        years = [2014, 2015, 2016]
        domains = ["laptops", "restaurants"]
        phases = ["Train", "Test"]
        
        for year in years:
            print(f"\nProcessing year: {year}")
            for domain in domains:
                print(f"  Processing domain: {domain}")
                for phase in phases:
                    try:
                        data = clean_data(year, phase, domain, force=force)
                        embeddings_dir = EmbeddingsDataset(year=year, device=device, phase=phase, domain=domain, empty_ok=True).dir
                        generate_embeddings(data, embeddings_dir, year, force=force)
                    except ValueError as e:
                        print(f"  Skipping {year} {domain} {phase}: {e}")
                    except FileNotFoundError as e:
                        print(f"  Skipping {year} {domain} {phase}: {e}")

if __name__ == "__main__":
    main()
