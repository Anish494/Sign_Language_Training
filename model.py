import torch
import torch.nn as nn
import math

# ── Positional Encoding ──────────────────────────────────────
# Same as your translation project!
# Gives the model sense of frame order (frame1, frame2... frame25)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=25, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Multi Head Attention ─────────────────────────────────────
# Same concept as your translation project!
# Each head learns different relationships between frames
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        
        # Q, K, V projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def attention(self, Q, K, V):
        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, V)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x = self.attention(Q, K, V)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(x)


# ── Encoder Block ────────────────────────────────────────────
# Attention → Add & Norm → FeedForward → Add & Norm
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self attention + residual connection
        x = self.norm1(x + self.dropout(self.attention(x)))
        # Feed forward + residual connection
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


# ── Full Transformer Classifier ──────────────────────────────
class SignLanguageTransformer(nn.Module):
    def __init__(
        self,
        input_dim=225,    # landmark features per frame
        d_model=128,      # transformer hidden size
        num_heads=4,      # attention heads
        num_layers=3,     # encoder blocks
        d_ff=256,         # feedforward dimension
        num_classes=22,   # number of signs
        max_seq_len=25,   # frames per video
        dropout=0.1
    ):
        super().__init__()
        
        # Project input landmarks to d_model dimensions
        # (like embedding layer in NLP)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        # Average all frame representations → classify
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 25, 225)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch, 25, 128)
        
        # Add positional encoding
        x = self.pos_encoding(x)      # (batch, 25, 128)
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)              # (batch, 25, 128)
        
        # Average across all frames
        x = x.mean(dim=1)             # (batch, 128)
        
        # Classify
        x = self.classifier(x)        # (batch, 22)
        
        return x


# ── Test Model ───────────────────────────────────────────────
if __name__ == "__main__":
    model = SignLanguageTransformer()
    
    # Dummy input: batch of 4 samples
    dummy = torch.randn(4, 25, 225)
    output = model(dummy)
    
    print("✅ Model works!")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     (4, 22)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")