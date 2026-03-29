import torch
import torch.nn.functional as F

from src.model import GCN
from src.data_loader import load_data


def train():
    data, dataset = load_data()

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    for epoch in range(200):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, data


def test(model, data):
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    model, data = train()
    test(model, data)