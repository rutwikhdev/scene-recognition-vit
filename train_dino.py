import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import load_vit_model
from src.args import get_args
from src.utils import set_random_seed, plot_confusion_matrix
from src.dataset import get_dataloaders
from torch.utils.tensorboard import SummaryWriter
from src.logger import Logger

import sys
import os.path as osp
import getpass

from datetime import datetime

torch.set_float32_matmul_precision("high") # enable tf32

def train(args):
    set_random_seed()

    # Define log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{getpass.getuser()}_{timestamp}.txt"
    sys.stdout = Logger(osp.join("logs", log_filename))
    print("\n==========\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    print(f"Model: dino classifier")
    print(f"Learning Rate: {args.lr}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir)

    print("Loading model")
    vit_backbone = load_vit_model("dinov2-embed").to(device)
    print(f"{args.model_name} Loaded")

    # Freeze backbone parameters
    for param in vit_backbone.parameters():
        param.requires_grad = False

    # Classifier head
    model = nn.Sequential(
        vit_backbone,  # Vision Transformer backbone
        nn.Linear(384, len(classes)),
    ).to(device)

    #gcc for neural nets, removes python overhead 
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    @torch.no_grad()
    def estimate_loss(eval_steps):
        model.eval()
        out = {}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = []
            for i, (x, y) in enumerate(loader):
                if i >= eval_steps:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        top5_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, preds = logits.topk(5, dim=1)
                correct += (preds[:, 0] == y).sum().item()
                top5_correct += sum([y[i] in preds[i] for i in range(len(y))])
                total += y.size(0)
        acc = correct / total
        top5 = top5_correct / total
        return {'acc': acc, 'top5': top5}

    print("\n==========Starting Training========\n")
    max_steps = 3500
    step = 0
    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype = torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
            optimizer.step()

            if step % 100 == 0:
                out = estimate_loss(100)
                print(f"Step {step}/{max_steps}  train_loss = {out['train']:.4f}  val_loss = {out['val']:.4f}")
                writer.add_scalar("Loss/train", out['train'], step)
                writer.add_scalar("Loss/val", out['val'], step)

            step += 1

    print("\n========Testing on validation set==========\n")
    out = evaluate()
    writer.add_scalar("Accuracy/Top1", out['acc'], step)
    writer.add_scalar("Accuracy/Top5", out['top5'], step)

    torch.save(model.state_dict(), "dino_classifier.pth")
    print("Model saved as dino_classifier.pth")


if __name__ == "__main__":
    args = get_args()
    print(args)
    train(args)