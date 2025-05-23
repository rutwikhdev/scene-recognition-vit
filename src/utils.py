import random

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from transformers import get_cosine_schedule_with_warmup


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_scheduler(scheduler, train_loader, optimizer, epochs=10):
    num_epochs = epochs
    train_steps_per_epoch = len(train_loader)
    total_training_steps = num_epochs * train_steps_per_epoch
    warmup_steps = int(0.1 * total_training_steps)

    if scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif scheduler == "cosinealr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=2e-5
        )
    elif scheduler == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0001,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
        )
    else:
        raise Exception(f"Learning rate scheduler {scheduler} not supported")

    return scheduler


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Convert one-hot encoded labels to class indices
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = labels.argmax(dim=1)

            outputs = model(pixel_values=images).logits
            _, top5 = outputs.topk(5, dim=1)

            correct_top1 += (top5[:, 0] == labels).sum().item()
            correct_top5 += sum([label in top for label, top in zip(labels, top5)])
            total += labels.size(0)

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    return top1_acc, top5_acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.
    Works with either hard or one-hot targets."""
    with torch.no_grad():
        if isinstance(output, (tuple, list)):
            output = output[0]

        # If target is one-hot (float and 2D), convert to class indices
        if target.ndim == 2 and target.dtype in (torch.float32, torch.float64):
            target = target.argmax(dim=1)

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, k]
        pred = pred.t()  # [k, B]

        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [k, B]

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc.item())
        return res


def plot_confusion_matrix(
    model, dataloader, class_names, device, save_path="confusion_matrix.png", save=True
):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 12))  # Increased size

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.show()
