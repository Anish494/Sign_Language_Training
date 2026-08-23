import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ── Apply mask ──────────────────────────────────────
        # mask shape: (batch, 1, 1, seq_len)
        # True  = padding position → set to -inf so softmax → 0
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        scores = torch.softmax(scores, dim=-1)

        # Handle NaN from all-padding rows
        scores = torch.nan_to_num(scores, nan=0.0)
        scores = self.dropout(scores)

        return torch.matmul(scores, V)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Expand mask for heads: (batch, 1, 1, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        x = self.attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention    = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention + residual
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        # Feed forward + residual
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class SignLanguageTransformer(nn.Module):
    def __init__(
        self,
        input_dim=291,    # updated for new dataset
        d_model=160,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        num_classes=52,   # updated for 52 signs
        max_seq_len=100,  # updated for new fixed length
        dropout=0.35
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding     = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def make_padding_mask(self, x, lengths):
        """
        Create mask where padding positions = True
        x shape:       (batch, seq_len, feat_dim)
        lengths shape: (batch,) — actual length of each sequence
        returns:       (batch, seq_len) bool tensor
        """
        batch_size, seq_len, _ = x.shape
        # Create position indices: (1, seq_len)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Compare with lengths: (batch, 1)
        lengths   = lengths.unsqueeze(1)
        # True where position >= actual length (padding)
        mask = positions >= lengths
        return mask  # (batch, seq_len)

    def forward(self, x, lengths=None):
        # x shape: (batch, seq_len, input_dim)

        # Create padding mask
        mask = None
        if lengths is not None:
            mask = self.make_padding_mask(x, lengths)

        # Project to d_model
        x = self.input_projection(x)   # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)       # (batch, seq_len, d_model)

        # Pass through encoder blocks with mask
        for block in self.encoder_blocks:
            x = block(x, mask)         # (batch, seq_len, d_model)

        # Masked average pooling
        # Only average over real frames, ignore padding
        if mask is not None:
            # mask: True = padding → invert for multiplication
            real_mask = (~mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
            x = (x * real_mask).sum(dim=1)             # sum real frames
            x = x / real_mask.sum(dim=1).clamp(min=1) # divide by real count
        else:
            x = x.mean(dim=1)

        # Classify
        x = self.classifier(x)         # (batch, num_classes)

        return x


if __name__ == "__main__":
    model = SignLanguageTransformer()

    # Test with variable length sequences
    batch  = torch.randn(4, 100, 291)
    # Simulate different actual lengths
    lengths = torch.tensor([100, 69, 45, 80])

    output = model(batch, lengths)

    print("✅ Model with masking works!")
    print(f"Input:   {batch.shape}")
    print(f"Lengths: {lengths}")
    print(f"Output:  {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")