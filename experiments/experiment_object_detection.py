from mlpipeline import Versions, MetricContainer, iterator
from mlpipeline.base import ExperimentABC, DataLoaderCallableWrapper
from mlpipeline.pytorch import (BaseTorchExperimentABC,
                                BaseTorchDataLoader,
                                DatasetFactory)
from mlpipeline.entities import ExecutionModeKeys, ExperimentModeKeys
from mlpipeline.utils import Datasets
import torch
import torchvision
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


class Experiment(BaseTorchExperimentABC):
    def setup_model(self):
        super().setup_model()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=self.current_version.num_classes)

        # pre-execution hook?
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model = self.model.cuda()
        self.lr_scheduler = None

    def _train_loop(self, input_fn, **kwargs):
        metricContainer = MetricContainer(["loss"])
        
        epochs = self.current_version.epocs
        self.log("Epochs: {}".format(epochs))
        epochs_end = epochs - self.epochs_params - 1
        if epochs_end < 1:
            epochs_end = 1

        self.log("Remaining epochs: {}".format(epochs_end))

        self.model.train()


        for epoch in iterator(range(self.epochs_params, self.epochs_params + epochs_end), 2):
            metricContainer.reset_epoch()
            for idx, (name, i) in iterator(enumerate(input_fn), 200):
                i = i.cuda()
                out = self.model(i)
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

    def _evaluate_loop(self, input_fn, **kwargs):
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

    def _post_execution_hook(self, mode, **kwargs):
        self.model.eval()
        out_location = Path("../exports") / self.current_version.name

        if not out_location.exists():
            out_location.mkdir(parents=True)

        self.copy_related_files(out_location)
            
        self.log("Exporting data to : " + str(out_location))
        torch.save({"state_dict": self.model.state_dict()}, out_location / "model.tch")

    def clean_experiment_dir(self, model_dir):
        self.log("CLEANING MODEL DIR")
        shutil.rmtree(model_dir, ignore_errors=True)


def add_version(name, dataset, flat, used_labels):
    print("*"*10, name)
    v.add_version(name,
                  dataloader=DataLoaderCallableWrapper(BaseTorchDataLoader,
                                                       datasets=Datasets("../data/generated/rico_temp.json",
                                                                         train_data_load_function=load_data,
                                                                         test_size=0.1,
                                                                         validation_size=0,
                                                                         used_labels=used_labels),
                                                       pytorch_dataset_factory=DatasetFactory(dataset, used_labels=used_labels),
                                                       batch_size=100),
                  epocs=75,
                  num_classes=len(used_labels),
                  )



v = Versions()
add_version("r1", SemanticConvModelDataSet, flat=False, used_labels=load_semantic_labels("../data/generated/semnantic_colors.json"))
EXPERIMENT = Experiment(v, allow_delete_experiment_dir=True)
