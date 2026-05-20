import math
import torch
import torch.nn as nn

def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe.unsqueeze(0)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=2, dropout=0.3, max_pos_len=5000):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        pe = _sinusoidal_positional_encoding(max_pos_len, d_model)
        self.register_buffer("pos_encoder", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        ) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, src_key_padding_mask=None): 
        x = self.embedding(x)
        seq_len = x.size(1)
        
        if seq_len > self.pos_encoder.shape[1]:
            x = x[:, :self.pos_encoder.shape[1], :]
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask[:, :self.pos_encoder.shape[1]] 
            seq_len = x.size(1)

        pos_emb = self.pos_encoder[:, :seq_len, :]
        x = x + pos_emb

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask_real = (~src_key_padding_mask).unsqueeze(-1).float() 
            sum_output = (output * mask_real).sum(dim=1)
            count = mask_real.sum(dim=1)
            embedding = sum_output / (count + 1e-8)
        else:
            embedding = output.mean(dim=1)

        logits = self.fc(embedding)
        return logits

    def get_attention(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        seq_len = x.size(1)
        
        if seq_len > self.pos_encoder.shape[1]:
            x = x[:, :self.pos_encoder.shape[1], :]
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask[:, :self.pos_encoder.shape[1]]
            seq_len = x.size(1)

        pos_emb = self.pos_encoder[:, :seq_len, :]
        x = x + pos_emb

        for i in range(len(self.transformer_encoder.layers) - 1):
            layer = self.transformer_encoder.layers[i]
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        last_layer = self.transformer_encoder.layers[-1]
        
        query = key = value = x
        attn_output, attn_weights = last_layer.self_attn(
            query, key, value, 
            key_padding_mask=src_key_padding_mask,
            need_weights=True 
        )
        
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)
            
        return attn_weights.cpu().detach()