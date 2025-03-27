from typing import Optional

import torch
from transformers import BertTokenizer, BertModel

from .bert_encoder import BertEncoder

tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model: BertModel = BertModel.from_pretrained("bert-base-uncased")
# print empty line after bert-base-uncased warnings
print()


class EmbeddingsLayer:
    def __init__(self, dense1, dense2, proj1, proj2,
                 device=torch.device('cpu')):
        super().__init__()

        self.device = device
        self.tokenizer: BertTokenizer = tokenizer
        self.model: BertModel = model.to(device)
        self.model.eval()
        self.encoder = BertEncoder(self.model, dense1=dense1, dense2=dense2, proj1=proj1, proj2=proj2)

    def forward(self, sentence: str, target_start: int, target_end: int, knowledge_layers) -> tuple[
        torch.Tensor, tuple[int, int], Optional[torch.Tensor]
    ]:
        sentence = f"[CLS] {sentence} [SEP]"
        target_start += 6
        target_end += 6

        # do not insert knowledge
        left_str = self.tokenizer.tokenize(sentence[0:target_start])
        target_str = self.tokenizer.tokenize(sentence[target_start:target_end])
        target_index_start = len(left_str) - 1
        target_index_end = target_index_start + len(target_str)

        tokens = self.tokenizer.tokenize(sentence)
        ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)], device=self.device)
        token_type_ids = torch.tensor([[0] * len(tokens)], device=self.device)

        initial_embeddings = self.model.embeddings.forward(input_ids=ids, token_type_ids=token_type_ids)

        embeddings: torch.Tensor = self.encoder(initial_embeddings, sentence=tokens, knowledge_layers=knowledge_layers)
        embeddings = embeddings[0][1:-1]

        return embeddings, (target_index_start, target_index_end), None
