import torch

from src.data_loader import load_data, split_train_nodes
from src.model import GCN
from src.client import train_local


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

    return acc


def compare_model_params(model_a, model_b):
    total_diff = 0.0
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        total_diff += torch.norm(param_a - param_b).item()
    return total_diff


def test_single_client_local_training():
    data, dataset = load_data()
    client_masks = split_train_nodes(data, num_clients=3, seed=42)

    global_model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes
    )

    acc_before = evaluate(global_model, data)
    print(f"Global model accuracy before local training: {acc_before:.4f}")

    local_model, num_samples = train_local(
        model=global_model,
        data=data,
        client_mask=client_masks[0],
        lr=0.01,
        weight_decay=5e-4,
        local_epochs=50
    )

    acc_after = evaluate(local_model, data)
    param_diff = compare_model_params(global_model, local_model)

    print(f"Client 0 sample count: {num_samples}")
    print(f"Local model accuracy after client 0 training: {acc_after:.4f}")
    print(f"Parameter difference between global and local model: {param_diff:.6f}")


if __name__ == "__main__":
    test_single_client_local_training()