from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def load_data():
    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=T.NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset