import argparse

from src.config import ALGORITHMS, PARTITION_MODES, TrainConfig
from src.experiments import (
    build_dirichlet_sweep,
    build_local_epochs_sweep,
    build_partition_sweep,
    build_strategy_comparison,
    run_experiment_suite,
)
from src.trainer import run_federated_training
from src.utils import ensure_dir, plot_training_history, print_table, save_records_to_csv, summarize_records


def parse_args():
    parser = argparse.ArgumentParser(description="Toy FGL experiments")
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "sweep_local_epochs",
            "compare_partitions",
            "sweep_dirichlet",
            "compare_strategies",
        ],
        default="train",
    )
    parser.add_argument("--partition_mode", choices=PARTITION_MODES, default="iid")
    parser.add_argument(
        "--partition_modes",
        nargs="*",
        choices=PARTITION_MODES,
        default=["iid", "soft_noniid", "hard_noniid"],
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="fedavg")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_epochs_list", type=int, nargs="*", default=[1, 5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--major_ratio", type=float, default=0.8)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--dirichlet_alphas", type=float, nargs="*", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--prox_mu", type=float, default=0.01)
    parser.add_argument("--shared_per_class", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--show_plot", action="store_true")
    return parser.parse_args()


def build_train_config(args, **overrides):
    params = {
        "num_rounds": args.num_rounds,
        "num_clients": args.num_clients,
        "local_epochs": args.local_epochs,
        "partition_mode": args.partition_mode,
        "seed": args.seeds[0],
        "major_ratio": args.major_ratio,
        "dirichlet_alpha": args.dirichlet_alpha,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_channels": args.hidden_channels,
        "algorithm": args.algorithm,
        "prox_mu": args.prox_mu,
        "shared_per_class": args.shared_per_class,
    }
    params.update(overrides)
    config = TrainConfig(**params)
    config.validate()
    return config


def save_train_outputs(result, output_dir: str, show_plot: bool):
    ensure_dir(output_dir)
    history_path = f"{output_dir}/history_{result.config.run_name}.csv"
    plot_path = f"{output_dir}/training_curve_{result.config.run_name}.png"
    summary_path = f"{output_dir}/summary_{result.config.run_name}.csv"

    save_records_to_csv(result.history, history_path)
    save_records_to_csv([result.summary_record()], summary_path)
    plot_training_history(result.history, filename=plot_path, show=show_plot)

    print(f"Saved history to {history_path}")
    print(f"Saved curve to   {plot_path}")
    print(f"Saved summary to {summary_path}")


def run_train_mode(args):
    config = build_train_config(args)
    result = run_federated_training(config, print_label_dist=True, verbose=True)
    save_train_outputs(result, args.output_dir, args.show_plot)
    print_table(
        [result.summary_record()],
        [
            "partition_mode",
            "strategy",
            "local_epochs",
            "best_global_acc",
            "final_global_acc",
            "public_nodes",
        ],
        title="Training Summary",
    )


def run_local_epochs_mode(args):
    base_config = build_train_config(args)
    configs = build_local_epochs_sweep(base_config, args.local_epochs_list, args.seeds)
    results = run_experiment_suite(configs, print_label_dist=False, verbose=True)
    records = [result.summary_record() for result in results]
    summary = summarize_records(records, group_keys=["local_epochs"], metric_key="final_global_acc")

    ensure_dir(args.output_dir)
    raw_path = f"{args.output_dir}/local_epochs_{base_config.partition_mode}_{base_config.strategy_name}_raw.csv"
    summary_path = (
        f"{args.output_dir}/local_epochs_{base_config.partition_mode}_{base_config.strategy_name}_summary.csv"
    )
    save_records_to_csv(records, raw_path)
    save_records_to_csv(summary, summary_path)

    print_table(
        summary,
        ["local_epochs", "mean_final_global_acc", "std_final_global_acc", "num_runs", "runs"],
        title="Local Epochs Summary",
    )
    print(f"Saved raw results to     {raw_path}")
    print(f"Saved summary results to {summary_path}")


def run_partition_mode(args):
    base_config = build_train_config(args)
    configs = build_partition_sweep(base_config, args.partition_modes, args.seeds)
    results = run_experiment_suite(configs, print_label_dist=False, verbose=True)
    records = [result.summary_record() for result in results]
    summary = summarize_records(records, group_keys=["partition_mode"], metric_key="final_global_acc")

    ensure_dir(args.output_dir)
    raw_path = f"{args.output_dir}/partitions_{base_config.strategy_name}_raw.csv"
    summary_path = f"{args.output_dir}/partitions_{base_config.strategy_name}_summary.csv"
    save_records_to_csv(records, raw_path)
    save_records_to_csv(summary, summary_path)

    print_table(
        summary,
        ["partition_mode", "mean_final_global_acc", "std_final_global_acc", "num_runs", "runs"],
        title="Partition Comparison",
    )
    print(f"Saved raw results to     {raw_path}")
    print(f"Saved summary results to {summary_path}")


def run_dirichlet_mode(args):
    base_config = build_train_config(args, partition_mode="dirichlet")
    configs = build_dirichlet_sweep(base_config, args.dirichlet_alphas, args.seeds)
    results = run_experiment_suite(configs, print_label_dist=False, verbose=True)
    records = [result.summary_record() for result in results]
    summary = summarize_records(records, group_keys=["dirichlet_alpha"], metric_key="final_global_acc")

    ensure_dir(args.output_dir)
    raw_path = f"{args.output_dir}/dirichlet_{base_config.strategy_name}_raw.csv"
    summary_path = f"{args.output_dir}/dirichlet_{base_config.strategy_name}_summary.csv"
    save_records_to_csv(records, raw_path)
    save_records_to_csv(summary, summary_path)

    print_table(
        summary,
        ["dirichlet_alpha", "mean_final_global_acc", "std_final_global_acc", "num_runs", "runs"],
        title="Dirichlet Alpha Summary",
    )
    print(f"Saved raw results to     {raw_path}")
    print(f"Saved summary results to {summary_path}")


def run_strategy_mode(args):
    base_config = build_train_config(args)
    configs = build_strategy_comparison(base_config, args.seeds, args.shared_per_class)
    results = run_experiment_suite(configs, print_label_dist=False, verbose=True)
    records = [result.summary_record() for result in results]
    summary = summarize_records(records, group_keys=["strategy"], metric_key="final_global_acc")

    ensure_dir(args.output_dir)
    raw_path = f"{args.output_dir}/strategies_{base_config.partition_mode}_raw.csv"
    summary_path = f"{args.output_dir}/strategies_{base_config.partition_mode}_summary.csv"
    save_records_to_csv(records, raw_path)
    save_records_to_csv(summary, summary_path)

    print_table(
        summary,
        ["strategy", "mean_final_global_acc", "std_final_global_acc", "num_runs", "runs"],
        title="Strategy Comparison",
    )
    print(f"Saved raw results to     {raw_path}")
    print(f"Saved summary results to {summary_path}")


def main():
    args = parse_args()

    if args.mode == "train":
        run_train_mode(args)
        return

    if args.mode == "sweep_local_epochs":
        run_local_epochs_mode(args)
        return

    if args.mode == "compare_partitions":
        run_partition_mode(args)
        return

    if args.mode == "sweep_dirichlet":
        run_dirichlet_mode(args)
        return

    if args.mode == "compare_strategies":
        run_strategy_mode(args)
        return


if __name__ == "__main__":
    main()
