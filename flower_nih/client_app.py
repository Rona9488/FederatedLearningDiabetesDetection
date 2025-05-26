"""flower-nih: A Flower / PyTorch app."""

import os
import json
import torch
import numpy as np
from datetime import datetime
from flwr.common import Context
from flwr.client import ClientApp, NumPyClient

from flower_nih.task import Net, get_weights, load_data, set_weights, test, train


def save_model(weights, round_number, training_dir):
    os.makedirs(training_dir, exist_ok=True)
    filepath = os.path.join(training_dir, f"model_round_{round_number}.npz")
    np.savez(filepath, *weights)
    print(f"✅ [Client] Model saved at {filepath}")


# Flower Client Definition
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        round_number = config.get("round_number", 0)
        training_id = config.get("training_id", "default_training")
        training_dir = os.path.join("received_global_models", f"training_{training_id}")

        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )

        new_weights = get_weights(self.net)
        save_model(new_weights, round_number, training_dir)

        return new_weights, len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)

        round_number = config.get("server_round", 0)
        training_id = config.get("training_id", "default_training")
        training_dir = os.path.join("received_global_models", f"training_{training_id}")

        result = {
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }

        metrics_path = os.path.join(training_dir, "metrics.json")
        os.makedirs(training_dir, exist_ok=True)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        else:
            metrics_data = {}

        metrics_data[str(round_number)] = result

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)

        print(f"✅ Evaluasi round {round_number} disimpan di {metrics_path}")

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


# client_fn definition
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(batch_size)

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# ClientApp
app = ClientApp(client_fn)
