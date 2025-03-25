import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    Description
    ----------
    A fairly straightforward RNN based classifier
    """
    def __init__(
        self, type, input_dim, hidden_dim, output_dim, num_layers, dropout
    ):
        assert type in ['LSTM', 'GRU']
        super(SimpleRNN, self).__init__()
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
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        print(f'x = {x.shape}')
        # pass through RNN and get the final hidden state
        _, hidden = self.rnn(x)
        print(f'hidden = {hidden.shape}')

        # stitch the last forward-looking and backward-looking hidden dims
        if self.num_layers == 1:
            hidden = torch.cat((hidden[0], hidden[1]), dim = 1)
        elif self.num_layers > 1:
            hidden = torch.cat((
                hidden[-2, :, :], 
                hidden[-1, :, :]
            ), dim = 1)

        print(f'hidden = {hidden.shape}')

        # pass through fully connected later
        out = self.dropout(hidden)
        out = self.fc(out)
        print(f'fc out = {out.shape}')
        out = self.softmax(out)
        print(f'softmax out = {out.shape}')

        return out

