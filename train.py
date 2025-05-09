import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.args import get_args
from src.dataset import get_dataloaders
from src.models import load_vit_model
from src.utils import set_random_seed, init_scheduler, evaluate, accuracy, plot_confusion_matrix
from src.logger import Logger

import sys
import os.path as osp
import getpass

from datetime import datetime


def train(args):
    set_random_seed()

    # Define log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"log_{getpass.getuser()}_{timestamp}"
    log_filename = f"{log_name}/{log_name}.txt"

    sys.stdout = Logger(osp.join("logs", log_filename))
    print("\n==========\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")

    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = load_vit_model(num_labels=len(class_names)).to(device)

    # setup optimizer, lr_scheduler, loss and tensorboard
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = init_scheduler(args.scheduler, train_loader, optimizer, args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    print("\n==========Starting Training========\n")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # log scheduler
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("LR", current_lr, epoch)
            print(f"Epoch [{epoch+1}/{args.epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.2f}, Accuracy: {accuracy(outputs, labels)}")

        acc1, acc5 = evaluate(model, val_loader, device)
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/Top1", acc1, epoch)
        writer.add_scalar("Accuracy/Top5", acc5, epoch)

    print("\n========Testing on validation set==========\n")
    acc1, acc5 = evaluate(model, val_loader, device)
    print("top-1: ", acc1)
    print("top-5: ", acc5)
    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("Accuracy/Top1", acc1, epoch)
    writer.add_scalar("Accuracy/Top5", acc5, epoch)

    torch.save(model.state_dict(), "vit_scene_classifier.pth")
    plot_confusion_matrix(model, val_loader, class_names, device)

if __name__ == "__main__":
    """
    To run use the following command

    python3 train.py \
    --batch-size 128
    --optimizer adamw
    --lr 0.0001
    --epoch 10
    """
    args = get_args()
    print(args)
    train(args)

