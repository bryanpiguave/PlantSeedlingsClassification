from dataclasses import dataclass

@dataclass
class Paths:
    data_dir: str

@dataclass
class Params:
    num_classes: int
    base_learning_rate: float
    initial_epochs: int
    fine_tuning_epochs: int
    batch_size: int
    img_width: int
    img_height: int

@dataclass
class SeedlingConfig:
    paths : Paths
    params: Params