from mlpipeline import Versions, MetricContainer
from mlpipeline.base import ExperimentABC, DataLoaderABC


class DataLoader(DataLoaderABC):
    def __init__(self):
        self.test_data = []
        self.train_data = []

    def get_test_input(self, **kwargs):
        return self.test_data

    def get_test_sample_count(self, **kwargs):
        return len(self.test_data)

    def get_train_input(self, **kwargs):
        return self.train_data

    def get_train_sample_count(self, **kwargs):
        return len(self.train_data)


class Experiment(ExperimentABC):
    def setup_model(self):
        pass

    def pre_execution_hook(self, **kwargs):
        pass

    def train_loop(self, input_fn, **kwargs):
        pass

    def evaluate_loop(self, input_fn, **kwargs):
        return MetricContainer()

    def export_model(self, **kwargs):
        pass

    def _export_model(self, export_dir):
        pass

    def post_execution_hook(self, **kwargs):
        pass


v = Versions()
v.add_version("Run-1")
EXPERIMENT = Experiment(v)
