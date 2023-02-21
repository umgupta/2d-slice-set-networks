""" 3D brain age model"""
from torch import nn

from src.arch.ukbb.brain_age_3d import Model as BrainAgeModel


class Model(BrainAgeModel):
    def init_weights(self):
        if self.initialization == "custom":
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out",
                        nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


def get_arch(*args, **kwargs):
    return {"net": Model(**kwargs)}
