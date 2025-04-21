# https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus

"""Utility functions that are specific to this ontology"""
# Copy the correct code from ontology_restaurant.py or ontology_laptop.py into this file

from rdflib import URIRef, Graph

NAMESPACE = "http://www.kimschouten.com/sentiment/restaurant"


def find_synonyms_for(resource: URIRef, ontology: Graph) -> list[str]:
    lex = [str(item[2]) for item in ontology.triples((resource, URIRef("#lex", NAMESPACE), None))]
    return lex


def find_uri_for(lex: str, ontology: Graph) -> URIRef | None:
    if lex == '"':
        return None

    result = ontology.query(f"""
                SELECT ?subject
                {{ ?subject restaurant:lex "{lex}" }}
                LIMIT 1
                """)
    for row in result:
        return row.subject
    return None