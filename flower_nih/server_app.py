"""flower-nih: A Flower / PyTorch server app."""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

from flwr.common import EvaluateIns, Metrics, Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flower_nih.task import Net, get_weights


# ===== Custom FedAvg with Model + Metric Saving =====
class FedAvgSaveModel(FedAvg):
    def __init__(self, training_id: str, save_dir: str = "checkpoints", **kwargs):
        self._training_id = training_id
        self.save_dir = os.path.join(save_dir, f"training_{training_id}")
        self.metrics_path = os.path.join(self.save_dir, "server_metrics.json")
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_data = {}
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            np.savez(os.path.join(self.save_dir, f"round_{server_round}.npz"), *ndarrays)
            print(f"✅ [Server] Model global disimpan: {self.save_dir}/round_{server_round}.npz")

        # Simpan metrik training (jika ada)
        if metrics:
            self._save_metrics(stage="fit", round_number=server_round, metrics=metrics)

        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Simpan metrik evaluasi (loss dan metrics)
        metric_entry = {"loss": loss if loss is not None else -1.0}
        metric_entry.update(metrics or {})
        self._save_metrics(stage="evaluate", round_number=server_round, metrics=metric_entry)

        return loss, metrics

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins = super().configure_fit(server_round, parameters, client_manager)

        updated_ins = []
        for client, ins in fit_ins:
            ins.config["round_number"] = server_round
            ins.config["training_id"] = self._training_id
            updated_ins.append((client, ins))

        return updated_ins

    def configure_evaluate(self, server_round, parameters, client_manager):
        configs = super().configure_evaluate(server_round, parameters, client_manager)

        updated_configs = []
        for client, evaluate_ins in configs:
            config = dict(evaluate_ins.config)
            config["server_round"] = server_round
            config["training_id"] = self._training_id

            updated_configs.append((client, EvaluateIns(evaluate_ins.parameters, config)))
        return updated_configs

    def _save_metrics(self, stage: str, round_number: int, metrics: dict):
        if stage not in self.metrics_data:
            self.metrics_data[stage] = {}
        self.metrics_data[stage][str(round_number)] = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics_data, f, indent=2)
        print(f"📈 [Server] Metrik '{stage}' round {round_number} disimpan di {self.metrics_path}")


# ===== Custom aggregation function =====
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0.0
    }


# ===== ServerApp entrypoint =====
def server_fn(context: Context):
    training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🟢 [Server] Memulai sesi training: training_{training_id}")

    # Inisialisasi model
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvgSaveModel(
        training_id=training_id,
        save_dir="/app/global_models",
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


# Create server app
app = ServerApp(server_fn=server_fn)
