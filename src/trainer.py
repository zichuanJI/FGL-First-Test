from dataclasses import dataclass

import torch

from src.client import train_local
from src.config import TrainConfig
from src.data_loader import get_client_masks, get_label_distribution, load_data
from src.model import GCN
from src.server import fedavg
from src.utils import set_seed


@dataclass
class TrainingResult:
    config: TrainConfig
    history: list[dict]
    initial_acc: float
    label_distributions: list[dict]
    public_nodes: int

    @property
    def best_global_acc(self) -> float:
        if not self.history:
            return self.initial_acc
        return max(item["global_acc"] for item in self.history)

    @property
    def final_global_acc(self) -> float:
        if not self.history:
            return self.initial_acc
        return self.history[-1]["global_acc"]

    def summary_record(self) -> dict:
        record = self.config.to_record()
        record.update(
            {
                "initial_acc": self.initial_acc,
                "best_global_acc": self.best_global_acc,
                "final_global_acc": self.final_global_acc,
                "public_nodes": self.public_nodes,
            }
        )
        return record


def evaluate(model, data, mask=None):
    eval_mask = data.test_mask if mask is None else mask
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[eval_mask] == data.y[eval_mask]).sum()
        acc = int(correct) / int(eval_mask.sum())
    return acc


def print_partition_report(label_distributions, public_nodes: int = 0):
    for client_id, distribution in enumerate(label_distributions):
        print(f"Client {client_id} label distribution: {distribution}")
    if public_nodes > 0:
        print(f"Shared anchor nodes added to every client: {public_nodes}")


def run_federated_training(
    config: TrainConfig,
    print_label_dist: bool = True,
    verbose: bool = True,
):
    config.validate()
    set_seed(config.seed)

    data, dataset = load_data()
    client_masks, public_mask = get_client_masks(
        data,
        partition_mode=config.partition_mode,
        num_clients=config.num_clients,
        seed=config.seed,
        major_ratio=config.major_ratio,
        dirichlet_alpha=config.dirichlet_alpha,
        shared_per_class=config.shared_per_class,
    )
    label_distributions = get_label_distribution(data, client_masks)
    public_nodes = int(public_mask.sum())

    if print_label_dist and verbose:
        print_partition_report(label_distributions, public_nodes=public_nodes)

    global_model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
    )

    initial_acc = evaluate(global_model, data)
    if verbose:
        print(f"Initial global accuracy: {initial_acc:.4f}")

    history = []
    for round_idx in range(config.num_rounds):
        if verbose:
            print(f"\n===== Federated Round {round_idx + 1} =====")

        local_models = []
        sample_counts = []
        client_test_accs = []
        train_losses = []

        for client_id, client_mask in enumerate(client_masks):
            local_model, stats = train_local(
                model=global_model,
                data=data,
                client_mask=client_mask,
                lr=config.lr,
                weight_decay=config.weight_decay,
                local_epochs=config.local_epochs,
                algorithm=config.algorithm,
                prox_mu=config.prox_mu,
            )
            client_test_acc = evaluate(local_model, data)

            if verbose:
                extra = ""
                if config.algorithm == "fedprox":
                    extra = f", prox_penalty={stats['last_prox_penalty']:.4f}"
                print(
                    f"Client {client_id}: num_samples={stats['num_samples']}, "
                    f"client_test_acc={client_test_acc:.4f}, "
                    f"last_loss={stats['last_loss']:.4f}{extra}"
                )

            local_models.append(local_model)
            sample_counts.append(stats["num_samples"])
            client_test_accs.append(client_test_acc)
            train_losses.append(stats["last_loss"])

        global_model = fedavg(local_models, sample_counts)
        global_acc = evaluate(global_model, data)
        avg_client_test_acc = sum(client_test_accs) / len(client_test_accs)
        avg_train_loss = sum(train_losses) / len(train_losses)

        if verbose:
            print(f"Round {round_idx + 1} global accuracy: {global_acc:.4f}")
            print(
                f"Round {round_idx + 1} average client-model test accuracy: "
                f"{avg_client_test_acc:.4f}"
            )

        history.append(
            {
                "round": round_idx + 1,
                "global_acc": global_acc,
                "avg_client_test_acc": avg_client_test_acc,
                "avg_train_loss": avg_train_loss,
            }
        )

    return TrainingResult(
        config=config,
        history=history,
        initial_acc=initial_acc,
        label_distributions=label_distributions,
        public_nodes=public_nodes,
    )
