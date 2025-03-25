"""
====================================
=== Hierarchical Attention Model ===
====================================

Inspired by the work of Tsaptsinos et al 2017 of using a hierarchical framework
for predicting song genre given lyrics.

NOTE: This architecture does not make sense when using DistilBERT, since we 
only have 1 vector per song, rather than 1 vector for word in each song.
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
        print(f'x = {x.shape}')
        rnn_out, _ = self.rnn(x)
        print(f'rnn_out = {rnn_out.shape}')
        attn_out = self.attention(rnn_out)
        print(f'attn_out = {attn_out.shape}')
        out = self.fc(attn_out)
        print(f'out = {out.shape}')

        return self.softmax(out)

