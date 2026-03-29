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


def test_one_round_federated_training():
    data, dataset = load_data()
    client_masks = split_train_nodes(data, num_clients=3, seed=42)

    global_model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes
    )

    acc_before = evaluate(global_model, data)
    print(f"Global model accuracy before federated round: {acc_before:.4f}")

    local_models = []
    sample_counts = []

    for i, client_mask in enumerate(client_masks):
        local_model, num_samples = train_local(
            model=global_model,
            data=data,
            client_mask=client_mask,
            lr=0.01,
            weight_decay=5e-4,
            local_epochs=50
        )

        local_acc = evaluate(local_model, data)
        print(f"Client {i}: num_samples={num_samples}, local_acc={local_acc:.4f}")

        local_models.append(local_model)
        sample_counts.append(num_samples)

    new_global_model = fedavg(local_models, sample_counts)

    acc_after = evaluate(new_global_model, data)
    print(f"Global model accuracy after federated round: {acc_after:.4f}")


if __name__ == "__main__":
    test_one_round_federated_training()