import torch
import torch.nn as nn

class NegativeCorrLoss(nn.Module):
    """Negative correlation loss function for PyTorch
    Credit to https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart/comments#1926321

    Precondition:
    y_true.mean(axis=1) == 0
    y_true.std(axis=1) == 1

    Returns:
    -1 = perfect positive correlation
    1 = totally negative correlation
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):

        my = torch.mean(preds, dim=1)
        my = torch.tile(torch.unsqueeze(my, dim=1), (1, targets.shape[1]))
        ym = preds - my
        r_num = torch.sum(torch.multiply(targets, ym), dim=1)
        r_den = torch.sqrt(
            torch.sum(torch.square(ym), dim=1) * float(targets.shape[-1])
        )
        r = torch.mean(r_num / r_den)
        return -r