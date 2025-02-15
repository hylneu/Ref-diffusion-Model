import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature):
        query = self.query(feature)
        key = self.key(feature)
        value = self.value(feature)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (feature.size(-1) ** 0.5)
        attention_probs = self.softmax(attention_scores)

        # Apply attention to the value
        attention_output = torch.matmul(attention_probs, value)

        return attention_output


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_feature, key_value_feature):
        query = self.query(query_feature)
        key = self.key(key_value_feature)
        value = self.value(key_value_feature)

        # Compute cross-attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key_value_feature.size(-1) ** 0.5)
        attention_probs = self.softmax(attention_scores)

        # Apply attention to the value
        attention_output = torch.matmul(attention_probs, value)

        return attention_output


class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024, num_attention_heads=8, num_layers=2):
        super(Image_adapter, self).__init__()

        # Attention layers
        self.self_attention_layers = nn.ModuleList([SelfAttention(hidden_size) for _ in range(num_layers)])
        self.cross_attention_layer = CrossAttention(hidden_size)

        # Adapter layer with dynamic layering
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature, cross_attention_input=None):
        # Self-attention layers: Dynamically stack based on the number of layers
        for attention_layer in self.self_attention_layers:
            feature = attention_layer(feature) + feature  # Residual connection after each attention layer

        # Cross-attention if provided
        if cross_attention_input is not None:
            feature = self.cross_attention_layer(feature, cross_attention_input) + feature

        # Pass through adapter
        out_feature = self.adapter(feature)
        return out_feature


# def cal_cos(text, img, cos):
#     a = text.mean(dim=1)
#     b = img.squeeze(0)
#     sim= cos(a, b).mean()
#
#     return sim





