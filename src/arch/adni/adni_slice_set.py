"""code for attention models"""
from torch import nn

from src.arch.ukbb.brain_age_slice_set import MRI2DSlice as BrainAge2DSlice


class MRI2DSlice(BrainAge2DSlice):
    def init_weights(self):
        if "resnet" in self.encoder_2d_name or self.initialization == "default":
            # only keep this init
            for k, m in self.named_modules():
                if isinstance(m, nn.Linear) and "regressor" in k:
                    m.bias.data.fill_(0.0)
        else:
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out",
                        nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear) and "regressor" in k:
                    m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


def get_arch(*args, **kwargs):
    return {"net": MRI2DSlice(*args, **kwargs)}
