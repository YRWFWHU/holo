from dataclasses import dataclass


@dataclass
class PropagationParams:
    wavelength: list
    prop_distance: float
    pixel_size: float
    resolution: list
    propagation_type: str


@dataclass
class Path:
    model_dir: str
    log_dir: str
    train_set: str
    valid_set: str
    test_set: str
    test_save_dir: str


@dataclass
class LearningModel:
    lr: float
    batch_size: int
    data_range: float
    num_workers: int
    num_devices: int
    max_epochs: int


@dataclass
class IterModel:
    lr: float
    batch_size: int
    data_range: float
    iter_num: int
    cuda: bool
    norm: bool


@dataclass
class DualNetConfig:
    prop_params: PropagationParams
    path: Path
    model_params: LearningModel


@dataclass
class HoloHDR:
    prop_params: PropagationParams
    model_params: IterModel
    path: Path
    num_subframe: int
    image_loss_weight: int
    laser_loss_weight: int
    variation_loss_weight: int
    anchor_wavelength: int
