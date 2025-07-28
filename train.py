"""Training script for models."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for the training of the model.

    Args:
        batch_size (int): The batch size.
        seq_len (int): The sequence length.
        max_steps (int): The maximum number of steps.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        device (str): The device to use.
        eval_interval (int): The interval to evaluate the model.
        save_interval (int): The interval to save the model.
    """

    batch_size: int = 128
    seq_len: int = 1024
    max_steps: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    device: str = "cuda"
    eval_interval: int = 100
    save_interval: int = 1000
    save_dir: str = "checkpoints"
    save_name: str = "nano-gpt"
    save_format: str = "pt"


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        pass

    def evaluate(self):
        pass
