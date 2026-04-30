"""Train source-only CNN1D baseline using only source-domain train and validation sets."""

import numpy as np
import torch
from sklearn.metrics import f1_score


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(
        targets.cpu().numpy(),
        preds.cpu().numpy(),
        average="macro",
        zero_division=0,
    )
    return accuracy, macro_f1


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    average_loss = total_loss / total_samples
    accuracy, macro_f1 = _compute_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
    )
    return average_loss, accuracy, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    average_loss = total_loss / total_samples
    accuracy, macro_f1 = _compute_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
    )
    return average_loss, accuracy, macro_f1

