from mlpipeline import Versions, MetricContainer, iterator
from mlpipeline.base import ExperimentABC, DataLoaderCallableWrapper
from mlpipeline.pytorch import (BaseTorchExperimentABC,
                                BaseTorchDataLoader,
                                DatasetFactory)
from mlpipeline.utils import Datasets
import torch

from dataloader import ConvModelDataSet, LinearModelDataSet, load_data
from models import LinearRicoAE, ConvRicoAE


class Experiment(BaseTorchExperimentABC):
    def setup_model(self):
        super().setup_model()
        self.model = self.current_version.model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()

        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()

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
                i = i.cuda()
                out = self.model(i)
                loss = self.criterion(out, i)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate_loop(self, input_fn, **kwargs):
        for i in input_fn:
            pass
        return MetricContainer()

    def export_model(self, **kwargs):
        pass

    def _export_model(self, export_dir):
        pass

    def post_execution_hook(self, **kwargs):
        pass


v = Versions()
# v.add_version("Run-1",
#               model=ConvRicoAE,
#               dataloader=DataLoaderCallableWrapper(BaseTorchDataLoader,
#                                                    datasets=Datasets("../data/generated/rico.json",
#                                                                      train_data_load_function=load_data,
#                                                                      test_size=0.05,
#                                                                      validation_size=0),
#                                                    pytorch_dataset_factory=DatasetFactory(ConvModelDataSet),
#                                                    batch_size=5),
#               epocs=100)

v.add_version("Run-1",
              model=LinearRicoAE,
              dataloader=DataLoaderCallableWrapper(BaseTorchDataLoader,
                                                   datasets=Datasets("../data/generated/rico.json",
                                                                     train_data_load_function=load_data,
                                                                     test_size=0.05,
                                                                     validation_size=0),
                                                   pytorch_dataset_factory=DatasetFactory(LinearModelDataSet),
                                                   batch_size=5),
              epocs=100)


EXPERIMENT = Experiment(v)
