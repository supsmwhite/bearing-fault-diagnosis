"""Train Deep CORAL and evaluate source-domain validation performance."""

import torch
from sklearn.metrics import f1_score


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == targets).float().mean().item()
    macro_f1 = f1_score(
        targets.cpu().numpy(),
        preds.cpu().numpy(),
        average="macro",
        zero_division=0,
    )
    return accuracy, macro_f1


def _covariance(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [batch_size, feature_dim].")
    if features.size(0) < 2:
        raise ValueError("coral_loss requires batch_size >= 2.")

    centered = features - features.mean(dim=0, keepdim=True)
    return centered.t().matmul(centered) / (features.size(0) - 1)


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.size(1) != target_features.size(1):
        raise ValueError("source and target feature dimensions must match.")

    source_cov = _covariance(source_features)
    target_cov = _covariance(target_features)
    feature_dim = source_features.size(1)
    return (source_cov - target_cov).pow(2).sum() / (4 * feature_dim * feature_dim)


def train_one_epoch_deep_coral(
    model,
    source_loader,
    target_loader,
    class_criterion,
    optimizer,
    device,
    coral_loss_weight=1.0,
):
    model.train()

    total_loss_sum = 0.0
    class_loss_sum = 0.0
    coral_loss_sum = 0.0
    total_source_samples = 0
    all_source_logits = []
    all_source_targets = []

    for (source_x, source_y), (target_x, _target_y) in zip(source_loader, target_loader):
        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        optimizer.zero_grad()

        source_logits, source_features = model(source_x)
        _, target_features = model(target_x)

        class_loss = class_criterion(source_logits, source_y)
        coral = coral_loss(source_features, target_features)
        total_loss = class_loss + coral_loss_weight * coral

        total_loss.backward()
        optimizer.step()

        source_batch_size = source_y.size(0)
        total_source_samples += source_batch_size
        total_loss_sum += total_loss.item() * source_batch_size
        class_loss_sum += class_loss.item() * source_batch_size
        coral_loss_sum += coral.item() * source_batch_size

        all_source_logits.append(source_logits.detach().cpu())
        all_source_targets.append(source_y.detach().cpu())

    source_logits = torch.cat(all_source_logits, dim=0)
    source_targets = torch.cat(all_source_targets, dim=0)
    source_accuracy, source_macro_f1 = _classification_metrics(source_logits, source_targets)

    return (
        total_loss_sum / total_source_samples,
        class_loss_sum / total_source_samples,
        coral_loss_sum / total_source_samples,
        source_accuracy,
        source_macro_f1,
    )


@torch.no_grad()
def evaluate_source_deep_coral(model, val_loader, class_criterion, device):
    model.eval()

    loss_sum = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        class_logits, _features = model(x)
        loss = class_criterion(class_logits, y)

        batch_size = y.size(0)
        loss_sum += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(class_logits.cpu())
        all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    accuracy, macro_f1 = _classification_metrics(logits, targets)

    return loss_sum / total_samples, accuracy, macro_f1
