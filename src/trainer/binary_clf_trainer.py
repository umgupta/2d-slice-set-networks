import numpy
import torch
from box import Box
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.trainer.regression_trainer import Trainer


class BinaryClfTrainer(Trainer):
    def validate(self, valid_loader, *args, **kwargs):
        """
            we expect validate to return loss and other aux losses that we want to log
        """
        preds = []
        ids = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                output = self.model(batch)
                preds.extend(output.y_pred.reshape(-1).cpu().tolist())
                if isinstance(batch[2], tuple):
                    ids.extend(list(batch[2]))
                else:
                    ids.extend(batch[2].reshape(-1).cpu().tolist())
                labels.extend(batch[1].reshape(-1).cpu().tolist())
        preds, ids, labels = numpy.array(preds), numpy.array(ids), numpy.array(labels)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(preds).float(), torch.tensor(labels).float(), reduction="mean"
        )
        # breakpoint()
        acc = accuracy_score(labels, (preds > 0) * 1)
        balanced_acc = balanced_accuracy_score(labels, (preds > 0) * 1)

        return loss, Box({"accuracy": acc, "balanced_acc": balanced_acc}), Box(
            {"preds": preds, "labels": labels, "ids": ids}
        )
