# ============================================================
#  Spiking Neural Network (SNN) — CIFAR-100 Classification
#  Framework: SNNTorch + PyTorch
#  Author: Pham Van Minh
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
import matplotlib.pyplot as plt
import numpy as np
import json

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE    = 128
NUM_EPOCHS    = 30
LEARNING_RATE = 3e-4
NUM_STEPS     = 50
BETA          = 0.90
NUM_CLASSES   = 100

# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

train_dataset = datasets.CIFAR100(root="./data", train=True,
                                   download=True, transform=transform_train)
test_dataset  = datasets.CIFAR100(root="./data", train=False,
                                   download=True, transform=transform_test)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")

# ─────────────────────────────────────────────
# 3. SNN MODEL DEFINITION
#    Conv1 → LIF → Conv2 → LIF → FC1 → LIF → FC2 → LIF
# ─────────────────────────────────────────────

spike_grad = surrogate.fast_sigmoid(slope=25)

class SpikingNet(nn.Module):
    def __init__(self, num_classes=100):
        super(SpikingNet, self).__init__()

        # Convolutional layers
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(64)
        self.lif_c1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        self.conv2  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(128)
        self.lif_c2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        self.pool   = nn.AvgPool2d(2)
        self.drop   = nn.Dropout(0.3)

        # Fully connected layers
        # After 2 pools: 32 → 16 → 8
        self.fc1  = nn.Linear(128 * 8 * 8, 512)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        self.fc2  = nn.Linear(512, num_classes)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x):
        # Initialise membrane potentials
        mem_c1 = self.lif_c1.init_leaky()
        mem_c2 = self.lif_c2.init_leaky()
        mem1   = self.lif1.init_leaky()
        mem2   = self.lif2.init_leaky()

        spike2_rec = []
        mem2_rec   = []

        for _ in range(NUM_STEPS):
            # Conv block 1
            cur_c1 = self.pool(self.bn1(self.conv1(x)))    # [batch, 64, 16, 16]
            spk_c1, mem_c1 = self.lif_c1(cur_c1, mem_c1)

            # Conv block 2
            cur_c2 = self.pool(self.bn2(self.conv2(spk_c1)))  # [batch, 128, 8, 8]
            spk_c2, mem_c2 = self.lif_c2(cur_c2, mem_c2)

            # Flatten
            x_flat = spk_c2.view(spk_c2.size(0), -1)       # [batch, 8192]
            x_flat = self.drop(x_flat)

            # FC block 1
            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.lif1(cur1, mem1)

            # FC block 2 (output)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spike2_rec), torch.stack(mem2_rec)


model = SpikingNet(num_classes=NUM_CLASSES).to(device)
print(f"\nSNN Architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. LOSS & OPTIMISER
# ─────────────────────────────────────────────

loss_fn   = SF.ce_rate_loss()
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimiser.zero_grad()
        spike_out, _ = model(data)
        loss         = loss_fn(spike_out, targets)
        loss.backward()
        optimiser.step()

        predicted  = spike_out.sum(dim=0).argmax(dim=1)
        correct   += (predicted == targets).sum().item()
        total     += targets.size(0)
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    return total_loss / len(train_loader), 100.0 * correct / total


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────

def evaluate():
    model.eval()
    correct      = 0
    total        = 0
    total_spikes = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spike_out, _ = model(data)
            predicted     = spike_out.sum(dim=0).argmax(dim=1)
            correct      += (predicted == targets).sum().item()
            total        += targets.size(0)
            total_spikes += spike_out.sum().item()

    accuracy   = 100.0 * correct / total
    avg_spikes = total_spikes / total
    return accuracy, avg_spikes


# ─────────────────────────────────────────────
# 7. RUN TRAINING
# ─────────────────────────────────────────────

train_losses = []
train_accs   = []
test_accs    = []
avg_spikes_list = []

print("\n" + "="*55)
print("  SNN TRAINING STARTED — CIFAR-100")
print("="*55)

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc    = train_one_epoch(epoch)
    test_acc, avg_spikes     = evaluate()
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    avg_spikes_list.append(avg_spikes)

    print(f"\nEpoch {epoch+1:02d}/{NUM_EPOCHS} Summary:")
    print(f"  Train Loss : {train_loss:.4f}")
    print(f"  Train Acc  : {train_acc:.2f}%")
    print(f"  Test Acc   : {test_acc:.2f}%")
    print(f"  Avg Spikes : {avg_spikes:.2f} per sample")
    print("-"*40)

print("\nSNN Training complete!")

# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(range(1, NUM_EPOCHS+1), train_losses, 'b-o', linewidth=2)
axes[0].set_title("SNN Training Loss (CIFAR-100)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(range(1, NUM_EPOCHS+1), train_accs, 'g-o', label="Train", linewidth=2)
axes[1].plot(range(1, NUM_EPOCHS+1), test_accs,  'r-o', label="Test",  linewidth=2)
axes[1].set_title("SNN Accuracy (CIFAR-100)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(range(1, NUM_EPOCHS+1), avg_spikes_list, 'm-o', linewidth=2)
axes[2].set_title("Avg Spikes per Sample")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Avg Spikes")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("snn_cifar100_results.png", dpi=150)
plt.show()
print("Plot saved to snn_cifar100_results.png")

# ─────────────────────────────────────────────
# 9. SAVE MODEL & RESULTS
# ─────────────────────────────────────────────

torch.save(model.state_dict(), "snn_cifar100_model.pth")
print("Model saved to snn_cifar100_model.pth")

results = {
    "train_losses"   : train_losses,
    "train_accs"     : train_accs,
    "test_accs"      : test_accs,
    "avg_spikes_list": avg_spikes_list
}
with open("snn_cifar100_results.json", "w") as f:
    json.dump(results, f)
print("Results saved to snn_cifar100_results.json")
