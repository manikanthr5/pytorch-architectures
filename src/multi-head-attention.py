import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.0, min_val=-1e-15):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)
        self.min_val = min_val

    def forward(self, queries, keys, values, mask=None):
        attention = torch.matmul(keys, queries.transpose(2, 3)) / self.temperature
        if mask is not None:
            attention = attention.masked_fill(mask == 0, self.min_val)
        attention = F.softmax(attention, axis=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, values)
        return output, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, model_dim, key_dim, val_dim, temperature, dropout=0.0, min_val=-1e-15, eps=1e-6):
        self.n_head = n_head
        self.key_dim = key_dim
        self.val_dim = val_dim

        self.query_matrix = nn.Linear(model_dim, self.n_head * key_dim, bias=False)
        self.key_matrix = nn.Linear(model_dim, self.n_head * key_dim, bias=False)
        self.value_matrix = nn.Linear(model_dim, self.n_head * key_dim, bias=False)
        self.fc = nn.Linear(self.n_head * key_dim, model_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature, dropout, min_val)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim, eps=eps)

    def forward(self, queries, keys, values, mask=None):
        batch_size, query_len, key_len, val_len = queries.size(0), queries.size(1), keys.size(1), values.size(1)

        residual = queries
        queries = self.query_matrix(queries).view(batch_size, query_len, self.n_head, self.key_dim)
        keys = self.key_matrix(keys).view(batch_size, query_len, self.n_head, self.key_dim)
        values = self.value_matrix(values).view(batch_size, val_len, self.n_head, self.val_dim)

        queries, keys, values = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        queries, attention = self.attention(queries, keys, values, mask=mask)
        queries = queries.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        queries = self.dropout(self.fc(queries))
        
        queries += residual

        queries = self.layer_norm(queries)
        return queries, attention 