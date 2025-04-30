import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os
import re

def load_sequence_from_npz(path: str, label: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads an .npz file with shape (50, 15), extracts last 6 columns,
    and returns a (1, 50, 6) tensor and a label tensor (1, 1).
    """
    data = np.load(path)
    key = data.files[0]  # assumes only one array in the file
    arr = data[key]      # shape: (50, 15)
    
    if arr.shape != (50, 15):
        raise ValueError(f"Expected (50, 15), got {arr.shape}")

    # Extract last 6 columns (Fx, Fy, Fz, Tx, Ty, Tz)
    wrench_data = arr[:, 9:]  # shape: (50, 6)
    
    # Convert to PyTorch tensor, add batch dim
    X_tensor = torch.tensor(wrench_data, dtype=torch.float32).unsqueeze(0)  # shape: (1, 50, 6)
    y_tensor = torch.tensor([[label]], dtype=torch.float32)                # shape: (1, 1)

    return X_tensor, y_tensor

def load_dataset_from_folder(folder: str, id: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads .npz files in the folder with optional ID in the filename.
    For example: id='(onpallet)' will match data(onpallet)*_*.npz.
    """
    X_list = []
    y_list = []

    pattern = f"data{id}*_*.npz"
    files = glob.glob(os.path.join(folder, pattern))

    for file in files:
        match = re.search(r'data.*_(\d)\.npz', os.path.basename(file))
        if not match:
            print(f"Skipping unrecognized filename: {file}")
            continue

        label = int(match.group(1))
        try:
            X_seq, y_seq = load_sequence_from_npz(file, label)
            X_list.append(X_seq)
            y_list.append(y_seq)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not X_list:
        raise RuntimeError(f"No valid .npz files found in '{folder}' with ID '{id}'")

    X = torch.cat(X_list, dim=0).to("cuda")
    y = torch.cat(y_list, dim=0).to("cuda")
    return X, y

#########################################################################33

# Parameters
sequence_length = 50  # e.g. 100 timesteps per motion
input_size = 6         # fx, fy, fz, tx, ty, tz
hidden_size = 32
num_layers = 1

files = 201
n_test = 20
X, y = load_dataset_from_folder("nndata", "")
X_train = X[:files-n_test]
y_train = y[:files-n_test]
X_test = X[-n_test:]
y_test = y[-n_test:]

files = 362
n_test = 50
X, y = load_dataset_from_folder("nndata", "(onpallet)")
X_train = torch.cat([X_train, X[:files-n_test]], dim=0)
y_train = torch.cat([y_train, y[:files-n_test]], dim=0)
X_test = torch.cat([X_test, X[-n_test:]], dim=0)
y_test = torch.cat([y_test, y[-n_test:]], dim=0)

# Model
class MotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)   # Only keep the final hidden state
        x = self.fc(hn[-1])         # Use the last layer's hidden state
        return x

model = MotionClassifier()
model.to("cuda")

# Training setup
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]).to("cuda"))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Inference on a new motion sequence
test_sequence = torch.randn(1, sequence_length, input_size).to("cuda") 

# Evaluation
model.eval()
conf = 0.8
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = (test_outputs > conf).float()
    correct = (predictions == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f"\nTest Accuracy on last 40 samples: {accuracy * 100:.2f}%")

with torch.no_grad():
    pred = model(X_test)
    pred_labels = (pred > conf).float()

print("Predicted:", pred_labels.squeeze().cpu().numpy())
print("Ground truth:", y_test.squeeze().cpu().numpy())

with torch.no_grad():
    pred = model(X_test)
    pred_labels = (pred > conf).float()
    true_labels = y_test

    tp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
    tn = ((pred_labels == 0) & (true_labels == 0)).sum().item()
    fp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
    fn = ((pred_labels == 0) & (true_labels == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
torch.save(model.state_dict(), "placement_verifier3.pt")