from dataclasses import asdict, dataclass


PARTITION_MODES = ("iid", "soft_noniid", "hard_noniid", "dirichlet")
ALGORITHMS = ("fedavg", "fedprox")


@dataclass(frozen=True)
class TrainConfig:
    num_rounds: int = 10
    num_clients: int = 3
    local_epochs: int = 10
    partition_mode: str = "iid"
    seed: int = 42
    major_ratio: float = 0.8
    dirichlet_alpha: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    hidden_channels: int = 16
    algorithm: str = "fedavg"
    prox_mu: float = 0.01
    shared_per_class: int = 0

    def validate(self):
        if self.partition_mode not in PARTITION_MODES:
            raise ValueError(f"Unknown partition_mode: {self.partition_mode}")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        if self.num_rounds < 0:
            raise ValueError("num_rounds must be >= 0")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be > 0")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be > 0")
        if not 0.0 <= self.major_ratio <= 1.0:
            raise ValueError("major_ratio must be in [0, 1]")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")
        if self.prox_mu < 0:
            raise ValueError("prox_mu must be >= 0")
        if self.shared_per_class < 0:
            raise ValueError("shared_per_class must be >= 0")

    @property
    def strategy_name(self) -> str:
        parts = [self.algorithm]
        if self.shared_per_class > 0:
            parts.append(f"shared{self.shared_per_class}")
        return "+".join(parts)

    @property
    def run_name(self) -> str:
        parts = [
            self.partition_mode,
            self.strategy_name,
            f"le{self.local_epochs}",
            f"seed{self.seed}",
        ]
        if self.partition_mode == "dirichlet":
            parts.append(f"alpha{self.dirichlet_alpha:g}")
        if self.algorithm == "fedprox":
            parts.append(f"mu{self.prox_mu:g}")
        return "_".join(parts)

    def to_record(self) -> dict:
        record = asdict(self)
        record["strategy"] = self.strategy_name
        record["run_name"] = self.run_name
        return record
