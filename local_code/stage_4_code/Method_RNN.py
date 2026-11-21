"""Custom recurrent classifier and the instructor's Method interface."""
import torch
from torch import nn
from local_code.base_class.method import method

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, cell='RNN', embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.recurrent = getattr(nn, cell)(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, token_ids):
        states, _ = self.recurrent(self.embedding(token_ids))
        mask = token_ids.ne(0).unsqueeze(-1)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(self.dropout(pooled))

class Method_RNN(method):
    def __init__(self, mName, mDescription, args):
        super().__init__(mName, mDescription)
        self.args = args

    def run(self):
        # Import here to keep model definitions independent of experiment execution.
        from local_code.stage_4_code.experiment import train
        return train(self.args, rows=self.data)
