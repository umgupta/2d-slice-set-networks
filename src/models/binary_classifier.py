import torch

from lib.base_model import Base as BaseModel


class BinaryClassifier(BaseModel):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        return self.net(batch[0].to(self.device))

    def loss(self, pred, batch, reduce=True):
        y = batch[1].to(self.device).float()
        N = y.shape[0]
        y = y.reshape(N, -1)
        y_pred = pred.y_pred.reshape(N, -1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred, y,
            reduction="none" if not
            reduce else "mean"
            )
        acc = ((y_pred > 0).long() == y.long()).float()
        if reduce:
            acc = acc.mean()
        return loss, {"accuracy": acc}, y_pred
