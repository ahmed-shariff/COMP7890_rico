from mlpipeline import Versions, MetricContainer, iterator
from mlpipeline.base import DataLoaderCallableWrapper
from mlpipeline.pytorch import (BaseTorchDataLoader,
                                DatasetFactory)
from mlpipeline.entities import ExecutionModeKeys, ExperimentModeKeys
from mlpipeline.utils import Datasets
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil

from dataloader import (ConvModelDataSet,
                        LinearModelDataSet,
                        load_data,
                        imshow_tensor,
                        SemanticConvModelDataSet,
                        FLAT_USED_LABELS,
                        load_semantic_labels)

from experiment import Experiment

from models import ConvRicoAE


class ExperimentImg(Experiment):
    def train_loop(self, input_fn, **kwargs):
        metricContainer = MetricContainer(["loss"])
        
        epochs = self.current_version.epocs
        self.log("Epochs: {}".format(epochs))
        epochs_end = epochs - self.epochs_params - 1
        if epochs_end < 1:
            epochs_end = 1

        self.log("Remaining epochs: {}".format(epochs_end))

        self.model.train()


        for epoch in iterator(range(self.epochs_params, self.epochs_params + epochs_end), 1):
            metricContainer.reset_epoch()
            import time
            ti = time.time()
            for idx, (name, i, oi) in iterator(enumerate(input_fn), 20):
                # print(ti - time.time())
                i = i.cuda()
                oi = oi.cuda()
                out = self.model(oi)
                loss = self.criterion(out, i)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metricContainer.loss.update(loss.item(), 1)

                if idx % 100 == 0:
                    imshow_tensor(out, i.shape)
                    out_string_step = "Epoch: {}  Step: {}".format(
                        epoch + 1,
                        idx + 1)
                    self.log("----------------------", log_to_file=False)
                    self.log(out_string_step, log_to_file=False)
                    metricContainer.log_metrics(log_to_file=False)
                    metricContainer.reset()
                ti = time.time()

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

        for idx, (name, i, oi) in iterator(enumerate(input_fn), 20):
            i = i.cuda()
            oi = oi.cuda()
            out = self.model(oi)
            loss = self.criterion(out, i)
            metricContainer.loss.update(loss.item(), 1)
        return metricContainer

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
        for name, i, oi in tqdm(iterator(self.dataloader.get_train_input(mode=ExecutionModeKeys.TEST), 1), total=self.dataloader.get_train_sample_count()):
            train_encodings[name] = self.model.encode(oi.cuda()).cpu().detach().numpy()

        with open(out_location / "train_encodings.npy", "wb") as f:
            np.save(f, train_encodings)

        self.log("Encoding test data")
        test_encodings = {}
        for name, i, oi in tqdm(iterator(self.dataloader.get_test_input(mode=ExecutionModeKeys.TEST), 1), total=self.dataloader.get_test_sample_count()):
            test_encodings[name] = self.model.encode(oi.cuda()).cpu().detach().numpy()

        with open(out_location / "test_encodings.npy", "wb") as f:
            np.save(f, test_encodings)


def add_version(name, model, dataset, flat, used_labels):
    v.add_version(name,
                  model=model,
                  dataloader=DataLoaderCallableWrapper(BaseTorchDataLoader,
                                                       datasets=Datasets("../data/generated/rico.json",
                                                                         train_data_load_function=load_data,
                                                                         test_size=0.1,
                                                                         validation_size=0,
                                                                         used_labels=used_labels),
                                                       pytorch_dataset_factory=DatasetFactory(dataset, used_labels=used_labels, flat=flat, return_img=True),
                                                       batch_size=100),
                  epocs=75)



v = Versions()
add_version("img-conv-flat", ConvRicoAE, ConvModelDataSet, flat=True, used_labels=FLAT_USED_LABELS)
add_version("img-conv-non-flat-semantic", ConvRicoAE, SemanticConvModelDataSet, flat=False, used_labels=load_semantic_labels("../data/generated/semnantic_colors.json"))
EXPERIMENT = ExperimentImg(v, allow_delete_experiment_dir=True)
