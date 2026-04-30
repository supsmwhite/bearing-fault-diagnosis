"""Train DANN and evaluate source-domain validation performance."""

import torch
from sklearn.metrics import f1_score


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple:
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == targets).float().mean().item()
    macro_f1 = f1_score(
        targets.cpu().numpy(),
        preds.cpu().numpy(),
        average="macro",
        zero_division=0,
    )
    return accuracy, macro_f1


def train_one_epoch_dann(
    model,
    source_loader,
    target_loader,
    class_criterion,
    domain_criterion,
    optimizer,
    device,
    domain_loss_weight=1.0,
):
    model.train()

    total_loss_sum = 0.0
    class_loss_sum = 0.0
    domain_loss_sum = 0.0
    total_source_samples = 0
    total_domain_correct = 0
    total_domain_samples = 0
    all_source_logits = []
    all_source_targets = []

    for (source_x, source_y), (target_x, _target_y) in zip(source_loader, target_loader):
        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        source_domain_y = torch.zeros(source_x.size(0), dtype=torch.long, device=device)
        target_domain_y = torch.ones(target_x.size(0), dtype=torch.long, device=device)

        optimizer.zero_grad()

        source_class_logits, source_domain_logits = model(source_x)
        _, target_domain_logits = model(target_x)

        class_loss = class_criterion(source_class_logits, source_y)
        source_domain_loss = domain_criterion(source_domain_logits, source_domain_y)
        target_domain_loss = domain_criterion(target_domain_logits, target_domain_y)
        domain_loss = source_domain_loss + target_domain_loss
        total_loss = class_loss + domain_loss_weight * domain_loss

        total_loss.backward()
        optimizer.step()

        source_batch_size = source_y.size(0)
        domain_batch_size = source_domain_y.size(0) + target_domain_y.size(0)
        total_source_samples += source_batch_size
        total_domain_samples += domain_batch_size
        total_loss_sum += total_loss.item() * source_batch_size
        class_loss_sum += class_loss.item() * source_batch_size
        domain_loss_sum += domain_loss.item() * source_batch_size

        source_domain_pred = torch.argmax(source_domain_logits.detach(), dim=1)
        target_domain_pred = torch.argmax(target_domain_logits.detach(), dim=1)
        total_domain_correct += (source_domain_pred == source_domain_y).sum().item()
        total_domain_correct += (target_domain_pred == target_domain_y).sum().item()

        all_source_logits.append(source_class_logits.detach().cpu())
        all_source_targets.append(source_y.detach().cpu())

    source_logits = torch.cat(all_source_logits, dim=0)
    source_targets = torch.cat(all_source_targets, dim=0)
    source_accuracy, source_macro_f1 = _classification_metrics(source_logits, source_targets)
    domain_accuracy = total_domain_correct / total_domain_samples

    return (
        total_loss_sum / total_source_samples,
        class_loss_sum / total_source_samples,
        domain_loss_sum / total_source_samples,
        source_accuracy,
        source_macro_f1,
        domain_accuracy,
    )


@torch.no_grad()
def evaluate_source(model, val_loader, class_criterion, device):
    model.eval()

    loss_sum = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        class_logits, _domain_logits = model(x)
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

