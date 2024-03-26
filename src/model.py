from src.logger import logging
from src.utils import Utils


class BasicModel:
    def __init__(self, weights_file: str, conf_threshold: float = 0.45) -> None:
        self.weights = Utils.get_data_dir().joinpath("weights").joinpath(weights_file)
        logging.info(
            "Checking Weights File: %s, Exists: %s",
            self.weights.name,
            self.weights.exists(),
        )
        
        if not self.weights.exists():
            logging.warning("Exitting Due To Weights File Does Not Exists")
            exit(1)
                    
        self.weights = self.weights.as_posix()

        self.device = ...
        self.model = ...
        self.conf_threshold: float = conf_threshold

    def pre_process(self, frame): ...

    def predict(self, frame, rate: float = 0.45): ...

    def post_process(self, outputs, frame, threshold: float = 0.4): ...

    def extract_boxes(self, boxes): ...
