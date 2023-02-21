"""code for attention models"""
import torch
from box import Box
from torch import nn

from src.arch.slice_cnn_utils import Encoder1_2D, MaxPool, MeanPool, PooledAttention, ResNetEncoder


class MRI2DSlice(nn.Module):
    def __init__(
            self, attn_num_heads, attn_dim, attn_drop=False, agg_fn="attention", in_channel=1,
            slice_dim=1, use_position_encoding=False, encoder_2d=None, resnet_module="resnet18",
            load_pretrained_encoder=False, initialization="custom", *args, **kwargs
    ):
        super(MRI2DSlice, self).__init__()

        self.input_dim = [(1, 109, 91), (91, 1, 91), (91, 109, 1)][slice_dim - 1]
        self.num_slices = [91, 109, 91][slice_dim - 1]

        self.initialization = initialization
        self.num_heads = attn_num_heads
        self.attn_dim = attn_dim
        self.attn_drop = attn_drop
        self.agg_fn = agg_fn
        self.slice_dim = slice_dim
        self.use_position_encoding = use_position_encoding
        self.encoder_2d_name = encoder_2d
        self.load_pretrained_encoder = load_pretrained_encoder
        self.resnet_module = resnet_module

        self.in_channel = in_channel
        self.encoder_2d = self.create_2d_encoder()

        if agg_fn == "attention":
            self.pooled_attention = PooledAttention(
                input_dim=self.num_heads * self.attn_dim,
                dim_v=self.num_heads * self.attn_dim,
                dim_k=self.num_heads * self.attn_dim,
                num_heads=self.num_heads, ln=True
            )
        elif agg_fn == "mean":
            self.pooled_attention = MeanPool()
        elif agg_fn == "max":
            self.pooled_attention = MaxPool()
        else:
            raise Exception("Invalid attention function")

        # Build regressor
        self.attn_post = nn.Linear(self.num_heads * self.attn_dim, 64)
        self.regressor = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))
        self.init_weights()

        # some precomputed things for creating inputs
        self.collation_indices = list(
            zip(*[list(range(i, self.num_slices)) for i in range(in_channel)])
        )

    def create_2d_encoder(self):
        if self.encoder_2d_name == "encoder1":
            return Encoder1_2D(
                self.num_heads * self.attn_dim, self.slice_dim, in_channel=self.in_channel,
                use_position_encoding=self.use_position_encoding, dropout=self.attn_drop,
                post_proc_conv=None, encoder_2d=None, post_position_encoding=None
            )

        if "resnet" in self.encoder_2d_name:
            return ResNetEncoder(
                self.num_heads * self.attn_dim, self.slice_dim, in_channel=self.in_channel,
                use_position_encoding=self.use_position_encoding, dropout=self.attn_drop,
                resnet_module=self.encoder_2d_name, pretrained=self.load_pretrained_encoder
            )

    def init_weights(self):
        if "resnet" in self.encoder_2d_name or self.initialization == "default":
            # only keep this init
            for k, m in self.named_modules():
                if isinstance(m, nn.Linear) and "regressor" in k:
                    m.bias.data.fill_(62.68)
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
                    m.bias.data.fill_(62.68)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        B, C, H, W, D = x.size()

        # remove channel dimension as it is 1.
        x = x.squeeze(1)
        if self.slice_dim == 1:
            # collation indices are list of tuple, so the "i" is a tuple and shape will be x,1,y,z
            new_input = torch.cat([x[:, i, :, :] for i in self.collation_indices], dim=0)
            encoding = self.encoder_2d(new_input)
            encoding = torch.cat(
                [i.unsqueeze(2) for i in torch.split(encoding, B, dim=0)],
                dim=2
            )
            # note: squeezing without proper arguments is bad because batch dim can be dropped
            encoding = encoding.squeeze(4).squeeze(3)
        elif self.slice_dim == 2:
            new_input = torch.cat([x[:, :, i, :] for i in self.collation_indices], dim=0)
            new_input = torch.swapaxes(new_input, 1, 2)
            encoding = self.encoder_2d(new_input)
            encoding = torch.cat(
                [i.unsqueeze(3) for i in torch.split(encoding, B, dim=0)],
                dim=3
            )
            # note: squeezing without proper arguments is bad because batch dim can be dropped
            encoding = encoding.squeeze(4).squeeze(2)
        elif self.slice_dim == 3:
            new_input = torch.cat([x[:, :, :, i] for i in self.collation_indices], dim=0)
            new_input = torch.swapaxes(new_input, 1, 3)
            new_input = torch.swapaxes(new_input, 2, 3)
            encoding = self.encoder_2d(new_input)
            encoding = torch.cat(
                [i.unsqueeze(4) for i in torch.split(encoding, B, dim=0)],
                dim=4
            )
            # note: squeezing without proper arguments is bad because batch dim can be dropped
            encoding = encoding.squeeze(3).squeeze(2)
        else:
            raise Exception("Invalid slice dim")

        # swap dims for input to attention
        encoding = encoding.permute((0, 2, 1))
        encoding, attention = self.pooled_attention(encoding)
        return encoding.squeeze(1), attention

    def forward(self, x):
        embedding, attention = self.encode(x)
        post = self.attn_post(embedding)
        y_pred = self.regressor(post)
        return Box({"y_pred": y_pred, "attention": attention})

    def get_attention(self, x):
        _, attention = self.encode(x)
        return attention


def get_arch(*args, **kwargs):
    return {"net": MRI2DSlice(*args, **kwargs)}
