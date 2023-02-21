import torch

from lib.base_model import Base as BaseModel


class Regression(BaseModel):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        return self.net(batch[0].to(self.device))

    def loss(self, pred, batch, reduce=True):
        # ret_obj = {}
        y = batch[1].to(self.device).float()
        N = y.shape[0]
        y = y.reshape(N, -1)
        y_pred = pred.y_pred.reshape(N, -1)
        loss = torch.nn.functional.mse_loss(y_pred, y, reduction="none").sum(dim=1)
        if reduce:
            loss = loss.mean()

        return loss, {}, {}
