import torch
import torch.nn as nn

class StackingClassifier(nn.Module):
    def __init__(self, model="data/best_verifier.pt", conf=0.9):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self.conf = conf
        self.load_state_dict(torch.load(model))
        self.to("cuda")
        self.eval()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return self.sigmoid(x)
    
    def predict(self, data):
        if data.shape != (50, 6):
            raise ValueError(f"Expected input shape (50, 6), got {data.shape}")

        # Convert to tensor and add batch dimension
        test_seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to("cuda")  # shape (1, 50, 6)

        self.eval()
        with torch.no_grad():
            pred = self(test_seq)
        return int(pred.item() > self.conf)