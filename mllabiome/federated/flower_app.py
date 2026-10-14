from __future__ import annotations

from pathlib import Path

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from .model import initial_parameters, train_local, validate_parameters


class _Client(NumPyClient):
    def __init__(self, X: np.ndarray, y: np.ndarray, context: Context):
        self.X = X
        self.y = y
        self.context = context

    def get_parameters(self, config):
        return initial_parameters(self.X.shape[1])

    def fit(self, parameters, config):
        settings = self.context.run_config
        updates = train_local(
            parameters, self.X, self.y,
            alpha=float(settings["alpha"]),
            eta0=float(settings["eta0"]),
            local_epochs=int(settings["local-epochs"]),
            random_state=int(settings["random-state"]),
        )
        if any(np.max(np.abs(array)) > float(settings["clipping-range"]) for array in updates):
            raise ValueError("Local parameters exceed the configured SecAgg+ clipping range.")
        return updates, len(self.y), {}

    def evaluate(self, parameters, config):
        raise RuntimeError("Client-side evaluation is disabled in SecAgg+ LODO runs.")


def _client_fn(context: Context):
    index = int(context.node_config["partition-id"])
    count = int(context.node_config["num-partitions"])
    data_dir = Path(str(context.run_config["data-dir"]))
    if index < 0 or index >= count:
        raise ValueError("Invalid Flower client partition ID.")
    with np.load(data_dir / f"client_{index}.npz", allow_pickle=False) as source:
        X = np.asarray(source["X"], dtype=np.float64)
        y = np.asarray(source["y"], dtype=np.int64)
    return _Client(X, y, context).to_client()


client_app = ClientApp(client_fn=_client_fn, mods=[secaggplus_mod])


class _SavingFedAvg(FedAvg):
    def __init__(self, output_path: Path, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.n_features = n_features

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            arrays = parameters_to_ndarrays(parameters)
            validate_parameters(arrays, self.n_features)
            np.savez(
                self.output_path,
                coef=arrays[0],
                intercept=arrays[1],
                round=np.asarray(int(server_round), dtype=np.int64),
            )
            print(f"mllabiome SecAgg+ round {server_round} aggregated and checkpointed", flush=True)
        return parameters, metrics


server_app = ServerApp()


@server_app.main()
def _server_main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    n_clients = int(cfg["num-clients"])
    shares = int(cfg["num-shares"])
    threshold = int(cfg["reconstruction-threshold"])
    if n_clients < 3 or not 3 <= shares <= n_clients or not 2 <= threshold <= shares:
        raise ValueError("Invalid SecAgg+ configuration or insufficient training clients.")
    n_features = int(cfg["num-features"])
    initial = ndarrays_to_parameters(initial_parameters(n_features))
    strategy = _SavingFedAvg(
        output_path=Path(str(cfg["output-path"])),
        n_features=n_features,
        fraction_fit=1.0,
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        fraction_evaluate=0.0,
        accept_failures=False,
        initial_parameters=initial,
    )
    print(f"mllabiome SecAgg+ server starting: {n_clients} clients, {cfg['num-server-rounds']} rounds", flush=True)
    legacy = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=int(cfg["num-server-rounds"])),
        strategy=strategy,
    )
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=shares,
            reconstruction_threshold=threshold,
            max_weight=float(cfg["max-weight"]),
            clipping_range=float(cfg["clipping-range"]),
        ),
    )
    workflow(grid, legacy)
