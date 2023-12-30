import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.datapipes.utils.common 
import torchtext
import os
import torchtext.vocab as vocab

#save_path = r'C:\Users\Asus\Documents\Surrey\Semester1\NLP\CourseWork-Code\PriyaAnalysis\glove'
#with open(os.path.join(save_path, 'glove.6B.100d.pt'), 'rb') as f:
#    vectors = torch.load(f)
#glove = torchtext.vocab.Vectors(name=os.path.join(save_path, 'glove.6B.100d.txt'), cache=save_path)
#glove.vectors = vectors
#glove = torchtext.vocab.Vectors(name=os.path.join(save_path, 'glove.6B.100d.txt'), cache=save_path, vectors='glove.6B.100d.pt')
glove = vocab.GloVe(name='6B', dim=100)

class GRU(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, word2idx):
        super(GRU, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        embedding_matrix, word_2_idx = self.create_embedding_matrix(embedding_dim, word2idx)

        self.embedding = nn.Embedding(len(word_2_idx), embedding_dim)        
        self.embedding.weight.data.copy_(embedding_matrix)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.long()
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        embeds = self.embedding(x)
        out, _ = self.gru(embeds, h0)
        out = out[:, -1,:]
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
            
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
    #             ,weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    #     return hidden
    
    def create_embedding_matrix(self, embedding_dim, word_2_idx):        
        embedding_matrix = nn.init.xavier_uniform_(torch.empty((len(word_2_idx), embedding_dim)))
        for word, idx in word_2_idx.items():
            if word in glove.stoi:
                embedding_matrix[idx] = glove[word]
            else:
                word_d = word

        return embedding_matrix, word_2_idx
