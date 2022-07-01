from dataclasses import dataclass

@dataclass
class Paths:
    log: str
    data: str

@dataclass
class Params:
    num_classes: int
    base_learning_rate: float
    initial_epochs: int
    fine_tuning_epochs: int
    batch_size: int

@dataclass
class SeedlingConfig:
    paths : Paths
    params: Params