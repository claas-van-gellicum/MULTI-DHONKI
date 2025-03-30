from .embeddings_layer import EmbeddingsLayer


def generate_OneEmbedding(embeddings_layer: EmbeddingsLayer, sentence, target_from: int, target_to: int, knowledge_layers):
    embeddings, (left_index, right_index), hops = embeddings_layer.forward(sentence, target_from, target_to, knowledge_layers = knowledge_layers)
    left_index = int(left_index)
    right_index = int(right_index)
    data = {'embeddings': embeddings, 'target_left_index': left_index, 'target_right_index': right_index, 'hops': hops}

    return data

