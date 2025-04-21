# Clone the OWL2Vec-Star repository  from [Owl2Vec*](https://github.com/KRR-Oxford/OWL2Vec-Star.git)
# Copy this file to the OWL2Vec-Star directory
# Set default.cfg file in the OWL2Vec-Star directory (enable reasoner)
# Initial folder data/embeddings/ontology_laptops/output/ and data/embeddings/ontology_restaurant/output/

from owl2vec_star import owl2vec_star
import os

os.makedirs("data/embeddings/ontology_laptops/output/", exist_ok=True)
os.makedirs("data/embeddings/ontology_restaurants/output/", exist_ok=True)

#Parameters:
# ontology_file
# config_file
# uri_doc
# lit_doc
# mix_doc
gensim_model = owl2vec_star.extract_owl2vec_model("data/raw/ontology_laptops.owl", "OWL2Vec_Star/default_laptops.cfg", True, True, True)

output_folder="data/embeddings/ontology_laptops/output/"

#Gensim format
gensim_model.save(output_folder+"ontology_laptops.embeddings")
#Txt format
gensim_model.wv.save_word2vec_format(output_folder+"ontology_laptops.embeddings.txt", binary=False)

#--------------------------------
#restaurant
gensim_model = owl2vec_star.extract_owl2vec_model("data/raw/ontology_restaurants.owl", "OWL2Vec_Star/default_restaurants.cfg", True, True, True)

output_folder="data/embeddings/ontology_restaurants/output/"

#Gensim format
gensim_model.save(output_folder+"ontology_restaurants.embeddings")
#Txt format
gensim_model.wv.save_word2vec_format(output_folder+"ontology_restaurants.embeddings.txt", binary=False)