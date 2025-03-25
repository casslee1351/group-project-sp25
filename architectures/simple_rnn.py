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

