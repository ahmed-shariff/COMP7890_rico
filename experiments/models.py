import torch
import torch.nn as nn
import torch.nn.functional as F


class RicoAE(nn.Module):
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


class LinearRicoAE(RicoAE):
    def __init__(self, ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features=9216, out_features=2048),
                                     nn.ReLU(),
                                     nn.Linear(in_features=2048, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(in_features=256, out_features=64),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(in_features=256, out_features=2048),
                                     nn.ReLU(),
                                     nn.Linear(in_features=2048, out_features=9216),
                                     nn.ReLU())

class ConvRicoAE(RicoAE):
    def __init__(self, ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Flatten(),
                                     nn.Linear(in_features=4608, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(in_features=256, out_features=64),
                                     nn.ReLU(),)

        self.decoder = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(in_features=256, out_features=4608),
                                     nn.ReLU(),
                                     nn.Unflatten(1, (32, 16, 9)),

                                     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),)
