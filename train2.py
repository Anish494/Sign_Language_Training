import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from model2 import SignLanguageTransformer

# ── Config ───────────────────────────────────────────────────
DATA_PATH   = r"D:\Sign Lang Landmarks\processed_resplit"
FIXED_LEN   = 100
FEAT_DIM    = 291
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 0.0001
NUM_CLASSES = 52
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Load class mapping ───────────────────────────────────────
with open(r"E:\Sign_Train_First\class_mapping.json") as f:
    sign_to_idx = json.load(f)

idx_to_sign = {v: k for k, v in sign_to_idx.items()}

# ── Dataset Class ────────────────────────────────────────────
class NSLDataset(Dataset):
    """
    Custom Dataset that:
    1. Loads .npy files on the fly
    2. Pads/truncates to FIXED_LEN
    3. Returns (sequence, length, label)
       length = actual frames before padding
    """
    def __init__(self, split_name):
        self.samples = []  # list of (file_path, label, actual_length)
        split_path   = os.path.join(DATA_PATH, split_name)

        for class_name in sorted(os.listdir(split_path)):
            if class_name not in sign_to_idx:
                continue

            label      = sign_to_idx[class_name]
            class_path = os.path.join(split_path, class_name)

            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                # Get actual length without loading full array
                try:
                    seq = np.load(fpath)
                    actual_len = min(seq.shape[0], FIXED_LEN)
                    self.samples.append((fpath, label, actual_len))
                except:
                    pass

        print(f"  {split_name}: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, actual_len = self.samples[idx]

        seq = np.load(fpath).astype(np.float32)

        # Pad or truncate
        seq_len = seq.shape[0]
        if seq_len >= FIXED_LEN:
            # Take middle frames
            start = (seq_len - FIXED_LEN) // 2
            seq   = seq[start:start + FIXED_LEN]
            actual_len = FIXED_LEN
        else:
            # Pad with zeros
            pad = np.zeros((FIXED_LEN - seq_len, FEAT_DIM), dtype=np.float32)
            seq = np.vstack([seq, pad])

        return (
            torch.FloatTensor(seq),          # (100, 291)
            torch.LongTensor([actual_len]),  # (1,) actual length
            torch.LongTensor([label])        # (1,) class label
        )


# ── Collate function ─────────────────────────────────────────
def collate_fn(batch):
    """Stack batch into tensors"""
    seqs    = torch.stack([b[0] for b in batch])          # (B, 100, 291)
    lengths = torch.cat([b[1] for b in batch])            # (B,)
    labels  = torch.cat([b[2] for b in batch])            # (B,)
    return seqs, lengths, labels


# ── Create Datasets and Loaders ──────────────────────────────
print("\nLoading datasets...")
train_dataset = NSLDataset("train")
val_dataset   = NSLDataset("val")
test_dataset  = NSLDataset("test")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ── Model, Loss, Optimizer ───────────────────────────────────
model = SignLanguageTransformer(
    input_dim   = FEAT_DIM,
    num_classes = NUM_CLASSES,
    max_seq_len = FIXED_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# ── Training Loop ────────────────────────────────────────────
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

print("\nStarting training...")
print("=" * 70)

for epoch in range(EPOCHS):

    # ── Train ────────────────────────────────────────────────
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for seqs, lengths, labels in train_loader:
        seqs    = seqs.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels  = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(seqs, lengths)     # pass lengths for masking
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc  = correct / total
    train_loss = total_loss / len(train_loader)

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    val_loss_total = 0
    val_correct    = 0
    val_total      = 0

    with torch.no_grad():
        for seqs, lengths, labels in val_loader:
            seqs    = seqs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels  = labels.to(DEVICE)

            outputs       = model(seqs, lengths)
            val_loss_total += criterion(outputs, labels).item()
            val_correct    += (outputs.argmax(1) == labels).sum().item()
            val_total      += labels.size(0)

    val_acc  = val_correct / val_total
    val_loss = val_loss_total / len(val_loader)

    # ── Scheduler ────────────────────────────────────────────
    scheduler.step(val_loss)

    # ── Save best model ──────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model2.pth")
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")

    # ── Log ──────────────────────────────────────────────────
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

# ── Save training history ─────────────────────────────────────
np.save("train_history2.npy", {
    "train_losses": train_losses,
    "val_losses":   val_losses,
    "train_accs":   train_accs,
    "val_accs":     val_accs
})

print(f"\n✅ Training complete!")
print(f"Best Val Accuracy: {best_val_acc:.4f}")