import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds, h0)        
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        #x_permuted = out.permute(0, 2, 1)
        #out = torch.mean(out, dim=1)
        return out
        
