import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, num_classes=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
