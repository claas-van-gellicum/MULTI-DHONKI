import torch.nn as nn
from rdflib import URIRef, Graph
import torch
import gensim
from .act_fun import gelu



class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """

    def __init__(self, hidden_size, feedforward_size, layer, currentLayerIndex, dense1, dense2, proj1, proj2):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = layer.intermediate.dense
        self.linear_2 = layer.output.dense
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.currentLayerIndex = currentLayerIndex
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Change depending on in which layer the knowledge gets added & change name of paths
        if self.currentLayerIndex in range(9,12):
            ontology_path = "data/raw/ontology.owl"
            self.ontology = Graph().parse(ontology_path)
            self.synonym_vectors = {}
            self.path_owl = "data/myontology2016/output"

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

    def collect_projectedknowledge(self, sentence):
        knowledge_vectors = []

        for word in sentence:
            uri = self.find_uri_for(lex=word, ontology=self.ontology)
            synonyms = None
            if uri:
                synonyms = self.find_synonyms_for(uri, self.ontology)
            if(synonyms is not None):
                knowledge_vectors.append(self.get_synonym_vectors(synonyms))
            else:
                knowledge_vectors.append(torch.zeros(1,100))

        knowledge_vectors = torch.cat(knowledge_vectors).to(self.device)
        return knowledge_vectors

    def get_synonym_vectors(self, synonyms):
        vectors = []
        model = gensim.models.Word2Vec.load(self.path_owl)

        for synonym in synonyms:
            if synonym in model.wv.index_to_key:
                iri_vector = model.wv.get_vector(synonym)
                vectors.append(torch.tensor(iri_vector))

        if vectors:
            vectors = torch.stack(vectors)
            vector = torch.mean(vectors, dim=0, keepdim=True)
            return vector

        return torch.zeros(1,100)

    @staticmethod
    def find_synonyms_for(resource: URIRef, ontology: Graph) -> list[str]:
        NAMESPACE = "http://www.kimschouten.com/sentiment/restaurant"
        lex = [str(item[2]) for item in ontology.triples((resource, URIRef("#lex", NAMESPACE), None))]
        return lex

    @staticmethod
    def find_uri_for(lex: str, ontology: Graph) -> URIRef | None:
        if lex == '"':
            return None

        result = ontology.query(f"""
                    SELECT ?subject
                    {{ ?subject restaurant1:lex "{lex}" }}
                    LIMIT 1
                    """)
        for row in result:
            return row.subject
        return None




