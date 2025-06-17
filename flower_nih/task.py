"""flower-nih: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# === PROMETHEUS METRICS SETUP ===
import threading
from prometheus_client import start_http_server, Gauge, REGISTRY

# Fungsi bantu untuk hindari duplikat metric
def get_or_create_gauge(name, description):
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        return Gauge(name, description)

TRAIN_DURATION = get_or_create_gauge("train_duration_seconds", "Training duration in seconds")
VAL_LOSS = get_or_create_gauge("val_loss", "Validation loss after training")
VAL_ACCURACY = get_or_create_gauge("val_accuracy", "Validation accuracy after training")
TEST_LOSS = get_or_create_gauge("test_loss", "Loss on test data")
TEST_ACCURACY = get_or_create_gauge("test_accuracy", "Accuracy on test data")

# Jalankan server Prometheus hanya sekali saat Streamlit reload
prom_port = int(os.environ.get("PROM_PORT", "9100"))
start_http_server(prom_port)
# ================================


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, input_size=10, hidden_size=32, num_classes=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



def load_data(batch_size: int, df: pd.DataFrame = None, train_split: float = 0.8):
    if df is None:
        data_path = os.getenv("DATA_PATH", "/app/data/data-1.csv") 
        df = pd.read_csv(data_path)

    # # Pilih hanya kolom yang diperlukan
    # selected_features = [
    #     "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity",
    #     "Fruits", "Veggies", "DiffWalk", "Sex", "Age", "Diabetes_01"
    # ]
    # df = df[selected_features]

    # Pisahkan fitur dan label
    X = df.iloc[:, :-1].values  # Semua kolom kecuali yang terakhir
    y = df.iloc[:, -1].values   # Hanya kolom terakhir

    # Konversi ke tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split train/test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Training untuk model MLP"""


    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total


        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

        val_loss, val_acc = test(net, valloader, device)

        VAL_LOSS.set(val_loss)
        VAL_ACCURACY.set(val_acc)

        print(f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        results = {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }

    return results


def validate(net, testloader, device):
    """Validasi model di test set"""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()  # Mode evaluasi

    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def predict_single(model, input_data):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.tensor([input_data], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
    return pred
