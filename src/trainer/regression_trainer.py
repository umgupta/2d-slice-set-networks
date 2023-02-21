import numpy
import torch
from box import Box

from lib.base_trainer import Trainer as BaseTrainer


class RegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        self.gradient_norm_clip = kwargs.pop("gradient_norm_clip")
        self.accumulation_steps = kwargs.pop("accumulation_steps")
        super(RegressionTrainer, self).__init__(*args, **kwargs)

    """Basic Trainer adapted for this repo"""

    def run_iteration(self, batch, training: bool = True, reduce: bool = True):
        """
            batch : batch of data, directly passed to model as is
            training: if training set to true else false
            reduce: whether to compute loss mean or return the raw vector form
        """
        pred = self.model(batch)
        loss, aux_loss, output = self.model.loss(pred, batch, reduce=reduce)

        if training:
            loss = loss / self.accumulation_steps
            loss.backward()
            if self.gradient_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm_clip)

            if (self.step + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss, aux_loss, output

    def on_train_end(self, train_loader, valid_loader, *args, **kwargs):
        # clear buffers
        self.optimizer.step()
        self.optimizer.zero_grad()

    def validate(self, valid_loader, *args, **kwargs):
        """
            we expect validate to return mean and other aux losses that we want to log
        """
        preds = []
        ids = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                output = self.model(batch)
                # breakpoint()
                preds.extend(output.y_pred.reshape(-1).cpu().tolist())
                if isinstance(batch[2], tuple):
                    ids.extend(list(batch[2]))
                else:
                    ids.extend(batch[2].reshape(-1).cpu().tolist())
                labels.extend(batch[1].reshape(-1).cpu().tolist())
        preds, ids, labels = numpy.array(preds), numpy.array(ids), numpy.array(labels)

        mse = ((preds - labels) ** 2).mean()
        mae = (numpy.abs(preds - labels)).mean()

        return mse, Box({"mse": mse, "mae": mae}), Box(
            {"preds": preds, "labels": labels, "ids": ids}
        )
