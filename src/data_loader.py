from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch


def load_data():
    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    return data, dataset


def split_train_nodes(data, num_clients=3, seed=42):
    """
    将原始 train_mask 对应的训练节点划分给多个 client。
    每个 client 得到一个布尔 mask，只在自己的训练节点上计算 loss。
    """
    torch.manual_seed(seed)

    # 取出所有训练节点的索引
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]

    # 打乱顺序
    perm = torch.randperm(train_idx.size(0))
    train_idx = train_idx[perm]

    # 平均分给不同 client
    chunks = torch.chunk(train_idx, num_clients)

    client_masks = []
    for chunk in chunks:
        mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
        mask[chunk] = True
        client_masks.append(mask)

    return client_masks