import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super(BiLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional = True)        
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_dim)
        embeds = self.embedding(x)
        out, _ = self.bilstm(embeds, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        #out = out[:, -1]
        #x_permuted = out.permute(0, 2, 1)
        #out = torch.mean(out, dim=1)
        return out
            
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_(),
    #                 weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_())
    #     return hidden
