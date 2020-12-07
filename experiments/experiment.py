from mlpipeline import Versions, MetricContainer, iterator
from mlpipeline.base import ExperimentABC, DataLoaderCallableWrapper
from mlpipeline.pytorch import (BaseTorchExperimentABC,
                                BaseTorchDataLoader,
                                DatasetFactory)
from dataloader import DataSet, load_data
from mlpipeline.utils import Datasets


class Experiment(BaseTorchExperimentABC):
    def setup_model(self):
        super().setup_model()

    def train_loop(self, input_fn, **kwargs):
        metricContainer = MetricContainer(["test"])
        
        epochs = self.current_version.epocs
        self.log("Epochs: {}".format(epochs))
        epochs_end = epochs - self.epochs_params - 1
        if epochs_end < 1:
            epochs_end = 1

        self.log("Remaining epochs: {}".format(epochs_end))

        for epoch in iterator(range(self.epochs_params, self.epochs_params + epochs_end, 1)):
            metricContainer.reset_epoch()
            for idx, i in iterator(enumerate(input_fn), 1):
                print(i)

    def evaluate_loop(self, input_fn, **kwargs):
        return MetricContainer()

    def export_model(self, **kwargs):
        pass

    def _export_model(self, export_dir):
        pass

    def post_execution_hook(self, **kwargs):
        pass


v = Versions()
v.add_version("Run-1",
              dataloader=DataLoaderCallableWrapper(BaseTorchDataLoader,
                                                   datasets=Datasets("../data/generated/rico.json",
                                                                     train_data_load_function=load_data,
                                                                     validation_size=0),
                                                   pytorch_dataset_factory=DatasetFactory(DataSet),
                                                   batch_size=5),
              epocs=100)
EXPERIMENT = Experiment(v)
