import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from model import SignLanguageTransformer

# ── Config ───────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 0.0001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── Load Data ────────────────────────────────────────────────
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val   = np.load("X_val.npy")
y_val   = np.load("y_val.npy")

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")

# ── Convert to Tensors ───────────────────────────────────────
X_train_t = torch.FloatTensor(X_train).to(DEVICE)
y_train_t = torch.LongTensor(y_train).to(DEVICE)
X_val_t   = torch.FloatTensor(X_val).to(DEVICE)
y_val_t   = torch.LongTensor(y_val).to(DEVICE)

# ── DataLoaders ──────────────────────────────────────────────
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ── Model, Loss, Optimizer ───────────────────────────────────
model     = SignLanguageTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=True
)

# ── Training Loop ────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ── Train ────────────────────────────────────────────────
    model.train()
    total_loss = 0
    correct    = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss    = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == batch_y).sum().item()

    train_acc  = correct / len(X_train_t)
    train_loss = total_loss / len(train_loader)

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss    = criterion(val_outputs, y_val_t).item()
        val_acc     = (val_outputs.argmax(1) == y_val_t).float().mean().item()

    # ── Scheduler ────────────────────────────────────────────
    scheduler.step(val_loss)

    # ── Save best model ──────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")

    # ── Print progress ───────────────────────────────────────
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

print(f"\n✅ Training complete!")
print(f"Best Val Accuracy: {best_val_acc:.4f}")