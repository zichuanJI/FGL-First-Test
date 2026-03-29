import torch

from src.data_loader import load_data, split_train_nodes
from src.model import GCN
from src.client import train_local
from src.server import fedavg


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

    return acc


def run_federated_training(num_rounds=10, num_clients=3, local_epochs=10):
    data, dataset = load_data()
    client_masks = split_train_nodes(data, num_clients=num_clients, seed=42)

    global_model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes
    )

    initial_acc = evaluate(global_model, data)
    print(f"Initial global accuracy: {initial_acc:.4f}")

    history = []

    for round_idx in range(num_rounds):
        print(f"\n===== Federated Round {round_idx + 1} =====")

        local_models = []
        sample_counts = []
        local_accs = []

        for i, client_mask in enumerate(client_masks):
            local_model, num_samples = train_local(
                model=global_model,
                data=data,
                client_mask=client_mask,
                lr=0.01,
                weight_decay=5e-4,
                local_epochs=local_epochs
            )

            local_acc = evaluate(local_model, data)
            print(f"Client {i}: num_samples={num_samples}, local_acc={local_acc:.4f}")

            local_models.append(local_model)
            sample_counts.append(num_samples)
            local_accs.append(local_acc)

        global_model = fedavg(local_models, sample_counts)
        global_acc = evaluate(global_model, data)

        avg_local_acc = sum(local_accs) / len(local_accs)

        print(f"Round {round_idx + 1} global accuracy: {global_acc:.4f}")
        print(f"Round {round_idx + 1} average local accuracy: {avg_local_acc:.4f}")

        history.append({
            "round": round_idx + 1,
            "global_acc": global_acc,
            "avg_local_acc": avg_local_acc,
        })

    return global_model, history


if __name__ == "__main__":
    import torch

    final_model, history = run_federated_training(
        num_rounds=10,
        num_clients=3,
        local_epochs=10
    )

    print("\n===== Training History =====")
    for item in history:
        print(
            f"Round {item['round']}: "
            f"global_acc={item['global_acc']:.4f}, "
            f"avg_local_acc={item['avg_local_acc']:.4f}"
        )