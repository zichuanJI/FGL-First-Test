from dataclasses import replace

from src.config import TrainConfig
from src.trainer import run_federated_training


def run_experiment_suite(configs, print_label_dist: bool = False, verbose: bool = True):
    results = []
    for config in configs:
        if verbose:
            print("\n" + "=" * 60)
            print(
                "Running config: "
                f"partition={config.partition_mode}, "
                f"strategy={config.strategy_name}, "
                f"local_epochs={config.local_epochs}, "
                f"seed={config.seed}"
            )
            if config.partition_mode == "dirichlet":
                print(f"Dirichlet alpha: {config.dirichlet_alpha}")
            print("=" * 60)

        result = run_federated_training(
            config,
            print_label_dist=print_label_dist,
            verbose=verbose,
        )
        results.append(result)

    return results


def build_local_epochs_sweep(base_config: TrainConfig, local_epochs_list, seeds):
    return [
        replace(base_config, local_epochs=local_epochs, seed=seed)
        for local_epochs in local_epochs_list
        for seed in seeds
    ]


def build_partition_sweep(base_config: TrainConfig, partition_modes, seeds):
    return [
        replace(base_config, partition_mode=partition_mode, seed=seed)
        for partition_mode in partition_modes
        for seed in seeds
    ]


def build_dirichlet_sweep(base_config: TrainConfig, alpha_list, seeds):
    return [
        replace(base_config, partition_mode="dirichlet", dirichlet_alpha=alpha, seed=seed)
        for alpha in alpha_list
        for seed in seeds
    ]


def build_strategy_comparison(base_config: TrainConfig, seeds, shared_per_class: int):
    strategies = [
        {"algorithm": "fedavg", "shared_per_class": 0},
        {"algorithm": "fedprox", "shared_per_class": 0},
        {"algorithm": "fedavg", "shared_per_class": shared_per_class},
        {"algorithm": "fedprox", "shared_per_class": shared_per_class},
    ]
    return [
        replace(base_config, seed=seed, **strategy)
        for strategy in strategies
        for seed in seeds
    ]
