from mlpipeline import Versions, MetricContainer, iterator
from mlpipeline.base import ExperimentABC, DataLoaderCallableWrapper
from mlpipeline.pytorch import (BaseTorchExperimentABC,
                                BaseTorchDataLoader,
                                DatasetFactory)
from mlpipeline.entities import ExecutionModeKeys, ExperimentModeKeys
from mlpipeline.utils import Datasets
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataloader import ConvModelDataSet, LinearModelDataSet, load_data
from models import LinearRicoAE, ConvRicoAE


class Experiment(BaseTorchExperimentABC):
    def setup_model(self):
        super().setup_model()
        self.model = self.current_version.model()

        # pre-execution hook?
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005)
        self.criterion = torch.nn.MSELoss()

        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()
        self.lr_scheduler = None

    def train_loop(self, input_fn, **kwargs):
        metricContainer = MetricContainer(["loss"])
        
        epochs = self.current_version.epocs
        self.log("Epochs: {}".format(epochs))
        epochs_end = epochs - self.epochs_params - 1
        if epochs_end < 1:
            epochs_end = 1

        self.log("Remaining epochs: {}".format(epochs_end))

        self.model.train()

        for epoch in iterator(range(self.epochs_params, self.epochs_params + epochs_end, 1)):
            metricContainer.reset_epoch()
            for idx, (name, i) in iterator(enumerate(input_fn), 20):
                i = i.cuda()
                out = self.model(i)
                loss = self.criterion(out, i)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metricContainer.loss.update(loss.item(), 1)

                if idx % self.logging_iteration == 0:
                    out_string_step = "Epoch: {}  Step: {}".format(
                        epoch + 1,
                        idx + 1)
                    self.log("----------------------", log_to_file=False)
                    self.log(out_string_step, log_to_file=False)
                    metricContainer.log_metrics(log_to_file=False)
                    metricContainer.reset()

            if epoch % 5 == 0:
                self.save_checkpoint(epoch)

            self.log("=========== Epoch Stats: ===========", log_to_file=False)
            self.log(out_string_step, log_to_file=True)
            metricContainer.log_metrics(metrics=None,
                                        log_to_file=True,
                                        complete_epoch=True,
                                        items_per_row=5,
                                        charachters_per_row=100,
                                        step=epoch + 1)
            self.log("=========== Epoch Ends =============", log_to_file=False)
            metricContainer.reset_epoch()

    def evaluate_loop(self, input_fn, **kwargs):
        metricContainer = MetricContainer(["loss"])
        self.model.eval()

        for idx, (name, i) in iterator(enumerate(input_fn), 20):
            i = i.cuda()
            out = self.model(i)
            loss = self.criterion(out, i)
            metricContainer.loss.update(loss.item(), 1)
        return metricContainer

    # def export_model(self, **kwargs):
    #     pass

    # def _export_model(self, export_dir):
    #     pass

    def post_execution_hook(self, mode, **kwargs):
        self.model.eval()
        out_location = Path("../exports") / self.current_version.name

        if not out_location.exists():
            out_location.mkdir(parents=True)

        self.copy_related_files(out_location)
            
        self.log("Exporting data to : " + str(out_location))
        torch.save({"state_dict": self.model.state_dict()}, out_location / "model.tch")

        self.log("Encoding train data")
        train_encodings = {}
        for name, i in tqdm(iterator(self.dataloader.get_train_input(mode=ExecutionModeKeys.TEST), 1), total=self.dataloader.get_train_sample_count()):
            train_encodings[name] = self.model.encode(i.cuda()).cpu().detach().numpy()

        with open(out_location / "train_encodings.npy", "wb") as f:
            np.save(f, train_encodings)

        self.log("Encoding test data")
        test_encodings = {}
        for name, i in tqdm(iterator(self.dataloader.get_test_input(mode=ExecutionModeKeys.TEST), 1), total=self.dataloader.get_test_sample_count()):
            test_encodings[name] = self.model.encode(i.cuda()).cpu().detach().numpy()

        with open(out_location / "test_encodings.npy", "wb") as f:
            np.save(f, test_encodings)


def add_version(name, model, dataset, flat):
    v.add_version(name,
                  model=model,
                  dataloader=DataLoaderCallableWrapper(BaseTorchDataLoader,
                                                       datasets=Datasets("../data/generated/rico.json",
                                                                         train_data_load_function=load_data,
                                                                         test_size=0.1,
                                                                         validation_size=0),
                                                       pytorch_dataset_factory=DatasetFactory(dataset, flat=flat),
                                                       batch_size=120),
                  epocs=500)



v = Versions()
add_version("linear-flat", LinearRicoAE, LinearModelDataSet, flat=True)
add_version("conv-flat", ConvRicoAE, ConvModelDataSet, flat=True)
add_version("linear-non-flat", LinearRicoAE, LinearModelDataSet, flat=False)
add_version("conv-non-flat", ConvRicoAE, ConvModelDataSet, flat=False)
EXPERIMENT = Experiment(v, allow_delete_experiment_dir=True)
