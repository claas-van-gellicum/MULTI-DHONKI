# https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus

"""Utility functions that are specific to this ontology"""
# Copy this code into ontology.py if you wish to use this ontology

from rdflib import URIRef, Graph, Namespace

# Update this to match your new namespace
NAMESPACE = "http://www.semanticweb.org/karoliina/ontologies/2017/4/laptop#"

# Create a namespace object
LAPTOP = Namespace(NAMESPACE)

# Property names - update these if they're different in your ontology
LEX_PROPERTY = "lex"  # The property name for lexical entries
PREFIX = "laptop"     # The prefix used in the ontology (e.g., laptop:lex)


def find_synonyms_for(resource: URIRef, ontology: Graph) -> list[str]:
    lex = [str(item[2]) for item in ontology.triples((resource, LAPTOP.lex, None))]
    return lex


def find_uri_for(lex: str, ontology: Graph) -> URIRef | None:
    if lex == '"':
        return None

    # First try to find the URI directly using the namespace
    for s, p, o in ontology.triples((None, LAPTOP.lex, lex)):
        return s

    # If not found, try the SPARQL query with proper namespace binding
    result = ontology.query("""
        PREFIX laptop: <http://www.semanticweb.org/karoliina/ontologies/2017/4/laptop#>
        SELECT ?subject
        WHERE {
            ?subject laptop:lex ?lex .
            FILTER(?lex = ?search_lex)
        }
        LIMIT 1
        """,
        initBindings={'search_lex': lex}
    )
    
    for row in result:
        return row.subject
    return None