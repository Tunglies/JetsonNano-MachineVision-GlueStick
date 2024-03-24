from pathlib import Path


class Utils:
    @classmethod
    def get_root_dir(self) -> Path:
        return Path(__file__).parent.parent

    @classmethod
    def get_data_dir(self) -> Path:
        return self.get_root_dir().joinpath("data")

    @classmethod
    def clean_garbages(self, *args):
        for item in args:
            del item