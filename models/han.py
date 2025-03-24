"""
====================================
=== Hierarchical Attention Model ===
====================================

Inspired by the work of Tsaptsinos et al 2017 of using a hierarchical framework
for predicting song genre given lyrics.
"""
import torch
from torch import nn 

class Attention(nn.Module):
    """
    Description
    ----------
    Self attention mechanism for the HAN
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias = False)

    def forward(self, x):
        # compute attention scores
        attention_scores = torch.tanh(self.attention_weights(x))

        # normalize
        attention_weights = torch.softmax(attention_scores, dim = 1)

        # compute weigthed sum of inputs
        output = (x * attention_weights).sum(dim = 1)

        return output

class HANClassifier(nn.Module):
    """
    Description
    ----------
    Hierarchical Attention Network for song genre classification
    """
    def __init__(self, input_dim = 768, hidden_dim = 256, output_dim = 10):
        super(HANClassifier, self).__init__()
        self.rnn = nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            batch_first = True,
            bidirectional = True
        )
        self.attention = Attention(hidden_dim * 2) # because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        attn_out = self.attention(rnn_out)
        out = self.fc(attn_out)

        return self.softmax(out)

