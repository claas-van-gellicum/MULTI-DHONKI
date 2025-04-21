import torch.nn as nn
from rdflib import URIRef, Graph, RDFS
import torch
import gensim
from .act_fun import gelu
import math


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """

    def __init__(self, hidden_size, feedforward_size, layer, domain, currentLayerIndex, dense1, dense2, proj1, proj2, ont_hops=0, gamma=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = layer.intermediate.dense
        self.linear_2 = layer.output.dense
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.currentLayerIndex = currentLayerIndex
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.domain = domain
        self.path_owl = f'data/embeddings/ontology_{self.domain}/output/ontology_{self.domain}.embeddings'
        self.ont_hops = ont_hops
        self.gamma = gamma

        if self.currentLayerIndex in range(9,12):
            ontology_path = f"data/raw/ontology_{self.domain}.owl"
            self.ontology = Graph().parse(ontology_path)
            self.synonym_vectors = {}

        self.dense1 = dense1
        self.dense2 = dense2
        self.projection_matrix_1 = proj1
        self.projection_matrix_2 = proj2

    def forward(self, x, sentence, knowledge_layers):
        inter = self.linear_1(x)

        if self.currentLayerIndex in knowledge_layers and sentence is not None:
            knowledge = self.collect_projectedknowledge(sentence = sentence)
            know1 = self.projection_matrix_1(knowledge)
            know1 = know1.unsqueeze(0)
            expanded_states = torch.cat([inter, know1], dim=-1)
            inter = self.dense1(expanded_states)

        inter = gelu(inter)
        output = self.linear_2(inter)

        if self.currentLayerIndex in knowledge_layers and sentence is not None:
            know2 = self.projection_matrix_2(knowledge)
            know2 = know2.unsqueeze(0)
            expanded_output = torch.cat([output, know2], dim=-1)
            output = self.dense2(expanded_output)
            output = self.layer_norm(output)

        return output

    def collect_projectedknowledge(self, sentence: list[str]):
        knowledge_vectors = []

        for word in sentence:
            uri = self.find_uri_for(lex=word, ontology=self.ontology, domain=self.domain)
            if uri:
                hop_vectors = self.collect_multi_hop_knowledge(uri=uri, current_hop=0)
                knowledge_vectors.append(hop_vectors)
            else:
                knowledge_vectors.append(torch.zeros(1,100))

        knowledge_vectors = torch.cat(knowledge_vectors).to(self.device)
        return knowledge_vectors

    def collect_multi_hop_knowledge(self, uri: URIRef, current_hop: int):
        hop_info = []  
        
        self._collect_hop_info(uri, current_hop, 0, hop_info)
        
        normalization_factor = sum(weight * count for weight, count, _ in hop_info)
        
        if normalization_factor == 0:  
            return torch.zeros(1,100)
        
        normalized_vector = torch.zeros(1,100).to(self.device)
        for weight, count, vector_sum in hop_info:
            if count > 0:  
                normalized_weight = weight / normalization_factor
                normalized_vector += normalized_weight * vector_sum
        
        return normalized_vector
        
    def _collect_hop_info(self, uri: URIRef, ont_hops: int, current_hop: int, hop_info: list):

        if current_hop > ont_hops or uri is None or not isinstance(uri, URIRef):
            return
        
        synonyms = self.find_synonyms_for(uri, self.ontology, self.domain)
        vector_sum, count = self.get_synonym_vectors(synonyms)
        
        weight = 1.0 if current_hop == 0 else math.exp(-(current_hop + self.gamma))
        
        if len(hop_info) <= current_hop:
            hop_info.append([weight, count, vector_sum])
        else:
            existing_weight, existing_count, existing_sum = hop_info[current_hop]
            hop_info[current_hop] = [existing_weight, existing_count + count, existing_sum + vector_sum]
        
        for subclass_uri, _, _ in self.ontology.triples((None, RDFS.subClassOf, uri)):
            if subclass_uri is None or not isinstance(subclass_uri, URIRef):
                continue
            
            self._collect_hop_info(subclass_uri, ont_hops, current_hop + 1, hop_info)
        
        for _, _, superclass_uri in self.ontology.triples((uri, RDFS.subClassOf, None)):
            if superclass_uri is None or not isinstance(superclass_uri, URIRef):
                continue
            
            self._collect_hop_info(superclass_uri, ont_hops, current_hop + 1, hop_info)

    def get_synonym_vectors(self, synonyms):
        vectors = []
        model = gensim.models.Word2Vec.load(self.path_owl)

        for synonym in synonyms:
            if synonym in model.wv.index_to_key:
                iri_vector = model.wv.get_vector(synonym)
                vectors.append(torch.tensor(iri_vector))

        if vectors:
            vectors = torch.stack(vectors)
            vector_sum = torch.sum(vectors, dim=0, keepdim=True)
            return vector_sum, len(vectors)

        return torch.zeros(1,100), 0  


    @staticmethod
    def find_synonyms_for(resource: URIRef, ontology: Graph, domain: str) -> list[str]:

        if domain == "restaurants":
            query_domain = "restaurant"
        elif domain == "laptops":
            query_domain = "laptop"
        else:
            query_domain = domain
            
        NAMESPACE = f"http://www.kimschouten.com/sentiment/{query_domain}"
        
        lex_uri = URIRef(f"{NAMESPACE}#lex")
        lex = [str(item[2]) for item in ontology.triples((resource, lex_uri, None))]
        return lex

    @staticmethod
    def find_uri_for(lex: str, ontology: Graph, domain: str) -> URIRef | None:
        if lex == '"':
            return None

        if domain == "restaurants":
            query_domain = "restaurant"
        elif domain == "laptops":
            query_domain = "laptop"
            
        NAMESPACE = "http://www.kimschouten.com/sentiment"
        
        result = ontology.query(f"""
                    PREFIX {query_domain}: <{NAMESPACE}/{query_domain}#>
                    SELECT ?subject
                    WHERE {{ ?subject {query_domain}:lex "{lex}" }}
                    LIMIT 1
                    """)
        for row in result:
            return row.subject
        return None




