from src.data_loader import load_data, split_train_nodes


def inspect_client_splits():
    data, dataset = load_data()
    client_masks = split_train_nodes(data, num_clients=3, seed=42)

    total_train_nodes = int(data.train_mask.sum())
    print(f"Total train nodes: {total_train_nodes}")

    total_from_clients = 0
    union_mask = None

    for i, mask in enumerate(client_masks):
        num_nodes = int(mask.sum())
        total_from_clients += num_nodes
        print(f"Client {i} train nodes: {num_nodes}")

        if union_mask is None:
            union_mask = mask.clone()
        else:
            overlap = (union_mask & mask).sum().item()
            print(f"Overlap with previous clients for client {i}: {overlap}")
            union_mask = union_mask | mask

    print(f"Sum of all client train nodes: {total_from_clients}")
    print(f"Union of all client train nodes: {int(union_mask.sum())}")


if __name__ == "__main__":
    inspect_client_splits()