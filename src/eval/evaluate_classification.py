"""Compute common evaluation metrics for classification models."""

import numpy as np
import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate_classification(model, dataloader, device):
    model.eval()

    y_true_batches = []
    y_pred_batches = []

    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_true_batches.append(y.cpu().numpy())
        y_pred_batches.append(preds)

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }

