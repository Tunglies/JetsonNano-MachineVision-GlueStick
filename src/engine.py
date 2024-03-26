from src.logger import logging
from src.model import BasicModel


class Engine(BasicModel):
    def __init__(self, weights_file: str, conf_threshold: float = 0.45) -> None:
        super().__init__(weights_file, conf_threshold)