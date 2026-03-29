import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_training_history(history, filename: str = "training_curve.png", show: bool = False):
    rounds = [item["round"] for item in history]
    global_acc = [item["global_acc"] for item in history]
    client_acc = [item["avg_client_test_acc"] for item in history]

    plt.figure()
    plt.plot(rounds, global_acc, label="Global Accuracy")
    plt.plot(rounds, client_acc, label="Avg Client-Model Test Accuracy")
    plt.xlabel("Federated Rounds")
    plt.ylabel("Accuracy")
    plt.title("Federated Training Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


def save_records_to_csv(records, filename: str):
    rows = list(records)
    if not rows:
        return

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            normalized = {
                key: (str(value) if isinstance(value, (list, tuple, dict)) else value)
                for key, value in row.items()
            }
            writer.writerow(normalized)


def summarize_records(records, group_keys, metric_key: str = "final_global_acc"):
    grouped = defaultdict(list)
    for record in records:
        key = tuple(record[group_key] for group_key in group_keys)
        grouped[key].append(record[metric_key])

    summary = []
    for key in sorted(grouped.keys()):
        values = grouped[key]
        mean_value = sum(values) / len(values)
        variance = 0.0
        if len(values) > 1:
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)

        item = {group_key: value for group_key, value in zip(group_keys, key)}
        item[f"mean_{metric_key}"] = mean_value
        item[f"std_{metric_key}"] = math.sqrt(variance)
        item["num_runs"] = len(values)
        item["runs"] = values
        summary.append(item)

    return summary


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(records, columns, title: str | None = None):
    if not records:
        print("No records to display.")
        return

    headers = [column for column in columns]
    widths = []
    for column in columns:
        values = [_format_value(record.get(column, "")) for record in records]
        widths.append(max(len(column), *(len(value) for value in values)))

    if title:
        print(f"\n===== {title} =====")

    header_line = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(header_line)
    for record in records:
        line = "  ".join(
            _format_value(record.get(column, "")).ljust(width)
            for column, width in zip(columns, widths)
        )
        print(line)
