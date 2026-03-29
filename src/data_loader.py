from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


MAJOR_CLASSES_DEFAULT: Dict[int, List[int]] = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5, 6],
}


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _mask_from_indices(data, indices: List[int]):
    mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
    if indices:
        mask[indices] = True
    return mask


def load_data(root: str = "data/Cora", name: str = "Cora"):
    dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    return data, dataset


def split_train_nodes(data, num_clients: int = 3, seed: int = 42):
    """IID-style split over training nodes."""
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    generator = _make_generator(seed)
    perm = torch.randperm(train_idx.numel(), generator=generator)
    shuffled_idx = train_idx[perm]

    return [
        _mask_from_indices(data, chunk.tolist())
        for chunk in torch.chunk(shuffled_idx, num_clients)
    ]


def split_train_nodes_noniid(data, num_clients: int = 3):
    """Hard Non-IID: each client gets disjoint label groups."""
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    labels = data.y
    num_classes = int(labels.max().item()) + 1
    classes_per_client = max(1, num_classes // num_clients)

    client_masks = []
    for i in range(num_clients):
        class_start = i * classes_per_client
        class_end = (i + 1) * classes_per_client
        if i == num_clients - 1:
            class_end = num_classes

        selected_classes = set(range(class_start, class_end))
        client_indices = [
            int(idx)
            for idx in train_idx.tolist()
            if int(labels[idx]) in selected_classes
        ]
        client_masks.append(_mask_from_indices(data, client_indices))

    return client_masks


def split_train_nodes_noniid_soft(
    data,
    num_clients: int = 3,
    major_ratio: float = 0.8,
    seed: int = 42,
    major_classes: Dict[int, List[int]] | None = None,
):
    """
    Soft Non-IID:
    each client owns most samples from its major classes,
    while the remaining samples are distributed to the other clients.
    """
    if major_classes is None:
        major_classes = MAJOR_CLASSES_DEFAULT

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    labels = data.y
    num_classes = int(labels.max().item()) + 1
    generator = _make_generator(seed)

    class_to_indices = defaultdict(list)
    for idx in train_idx.tolist():
        class_to_indices[int(labels[idx])].append(idx)

    for class_id, indices in class_to_indices.items():
        perm = torch.randperm(len(indices), generator=generator).tolist()
        class_to_indices[class_id] = [indices[i] for i in perm]

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        indices = class_to_indices[class_id]
        if not indices:
            continue

        owners = [i for i in range(num_clients) if class_id in major_classes.get(i, [])]
        if not owners:
            for client_id, chunk in enumerate(torch.chunk(torch.tensor(indices), num_clients)):
                client_indices[client_id].extend(chunk.tolist())
            continue

        owner = owners[0]
        major_n = int(len(indices) * major_ratio)
        client_indices[owner].extend(indices[:major_n])

        rest = indices[major_n:]
        other_clients = [i for i in range(num_clients) if i != owner]
        if rest and other_clients:
            for client_id, chunk in zip(other_clients, torch.chunk(torch.tensor(rest), len(other_clients))):
                client_indices[client_id].extend(chunk.tolist())

    return [_mask_from_indices(data, indices) for indices in client_indices]


def split_train_nodes_dirichlet(
    data,
    num_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 1,
):
    """Sample client partitions with a Dirichlet distribution over classes."""
    rng = np.random.default_rng(seed)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    labels = data.y.tolist()
    num_classes = int(data.y.max().item()) + 1

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = [idx for idx in train_idx if labels[idx] == class_id]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(alpha=[alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(np.array(class_indices), split_points)

            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_size:
            break

    return [_mask_from_indices(data, indices) for indices in client_indices]


def build_public_mask(data, shared_per_class: int = 0, seed: int = 42):
    mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
    if shared_per_class <= 0:
        return mask

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    labels = data.y
    num_classes = int(labels.max().item()) + 1
    generator = _make_generator(seed + 10_007)

    for class_id in range(num_classes):
        class_idx = train_idx[labels[train_idx] == class_id]
        if class_idx.numel() == 0:
            continue
        perm = torch.randperm(class_idx.numel(), generator=generator)
        selected = class_idx[perm][: min(shared_per_class, class_idx.numel())]
        mask[selected] = True

    return mask


def apply_public_mask(client_masks, public_mask):
    if int(public_mask.sum()) == 0:
        return client_masks
    return [mask | public_mask for mask in client_masks]


def get_client_masks(
    data,
    partition_mode: str,
    num_clients: int = 3,
    seed: int = 42,
    major_ratio: float = 0.8,
    major_classes: Dict[int, List[int]] | None = None,
    dirichlet_alpha: float = 0.5,
    shared_per_class: int = 0,
):
    if partition_mode == "iid":
        client_masks = split_train_nodes(data, num_clients=num_clients, seed=seed)
    elif partition_mode == "hard_noniid":
        client_masks = split_train_nodes_noniid(data, num_clients=num_clients)
    elif partition_mode == "soft_noniid":
        client_masks = split_train_nodes_noniid_soft(
            data,
            num_clients=num_clients,
            major_ratio=major_ratio,
            seed=seed,
            major_classes=major_classes,
        )
    elif partition_mode == "dirichlet":
        client_masks = split_train_nodes_dirichlet(
            data,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partition_mode: {partition_mode}")

    public_mask = build_public_mask(data, shared_per_class=shared_per_class, seed=seed)
    client_masks = apply_public_mask(client_masks, public_mask)
    return client_masks, public_mask


def get_label_distribution(data, client_masks) -> List[Dict[int, int]]:
    distributions = []
    for mask in client_masks:
        labels = data.y[mask].tolist()
        distributions.append(dict(sorted(Counter(labels).items())))
    return distributions
