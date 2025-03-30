# https://github.com/autoliuweijie/K-BERT
# -*- encoding:utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn
from .layer_norm import LayerNorm
from .position_ffn import PositionwiseFeedForward
from .multi_headed_attn import MultiHeadedAttention


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """

    def __init__(self, args, layer, currentLayerIndex, dense1,dense2,proj1,proj2):
        super(TransformerLayer, self).__init__()
        self.currentLayerIndex =currentLayerIndex

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.dropout, layer
        )

        self.layer_norm_1 = LayerNorm(args.hidden_size, layer.attention.output.LayerNorm)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size, args.feedforward_size, layer, currentLayerIndex= self.currentLayerIndex, dense1=dense1, dense2=dense2, proj1=proj1, proj2=proj2
        )

        self.layer_norm_2 = LayerNorm(args.hidden_size, layer.output.LayerNorm)

    def forward(self, hidden, sentence, knowledge_layers, vm: Optional[torch.Tensor] = None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            vm: [seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.self_attn(hidden, hidden, hidden, vm)
        inter = self.layer_norm_1(inter + hidden)
        output = self.feed_forward(inter, sentence = sentence, knowledge_layers = knowledge_layers)
        output = self.layer_norm_2(output + inter)
        return output
