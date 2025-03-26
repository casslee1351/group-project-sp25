import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilBertRNN(nn.Module):
    """
    Description
    ----------
    A fairly straightforward RNN based classifier
    """
    def __init__(
        self, type, input_dim, hidden_dim, output_dim, num_layers, dropout
    ):
        assert type in ['LSTM', 'GRU']
        super(DistilBertRNN, self).__init__()
        self.type = type
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        if type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size = input_dim,
                hidden_size = hidden_dim,
                batch_first = True,
                bidirectional = True,
                num_layers = num_layers,
                dropout = dropout
            )
        elif type == 'GRU':
            self.rnn = nn.GRU(
                input_size = input_dim,
                hidden_size = hidden_dim,
                batch_first = True,
                bidirectional = True,
                num_layers = num_layers,
                dropout = dropout
            )
        
        self.dropout = nn.Dropout(p = dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim = -1) # apply along last dimension

    def forward(self, x):
        # ensure input has seq_len = 1 for the RNN
        # x: (batch_size, 1, input_dim)
        x = x.unsqueeze(1) # (batch_size, 1, input_dim)

        # pass through RNN and get the final hidden state
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        _, hidden = self.rnn(x)

        # stitch the last forward-looking and backward-looking hidden dims
        if self.num_layers == 1:
            # implement!
            print(f'hidden = {hidden.shape}')
            assert True == False, 'TODO: Implement this section'
        elif self.num_layers > 1:
            ## hidden: (batch_size, hidden_dim * 2)
            forward_hidden = hidden[-2, :, :] # last layers forward hidden state
            backward_hidden = hidden[-1, :, :] # last layers backward hidden state
            hidden = torch.cat((forward_hidden, backward_hidden), dim = -1)

        # pass through fully connected later
        ## fc out: (batch_size, output_dim)
        out = self.fc(hidden)
        out = self.softmax(out)

        return out

class DistilBertDeepTransformer(nn.Module):
    """
    Description
    ----------
    This model incorporates attention elements as made famous by the 
    Transformer model. 
    """
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim, num_layers, dropout):
        super(DistilBertDeepTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.num_layers = num_layers 
        self.dropout = dropout 

        # attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim = input_dim,
                num_heads = num_heads,
                batch_first = True
            )
            for _ in range(num_layers)
        ])

        # feed-forward layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout()
            )
            for _ in range(num_layers)
        ])

        self.norm_layers_attn = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.norm_layers_ffn = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])

        # fully connected layers
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(input_dim, output_dim)

        # # dropout for training improvement
        # self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # add a sequence dimension since we are using DistilBERT CLS Embedding
        # x: (batch_size, 1, 768)
        x = x.unsqueeze(1)

        # pass through attention layers
        # for attn_layer, norm_layer in zip(self.attention_layers, self.norm_layers):
        #     attn_out, _ = attn_layer(x, x, x) # self attention
        #     x = norm_layer(attn_out + x) # residual connection and normalization
        for i in range(self.num_layers):
            # multi-head self attention
            attn_out, _ = self.attention_layers[i](x, x, x)
            x = self.norm_layers_attn[i](attn_out + x) # resid connect and normalize

            # feed-forward network
            ffn_out = self.ffn_layers[i](x)
            x = self.norm_layers_ffn[i](ffn_out + x) # resid connect and normalize.

        # remove sequence dimension
        # x: (batch_size, 768)
        
        x = x.squeeze(1)

        # full conected layers
        x = self.fc(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim = -1)

        return x