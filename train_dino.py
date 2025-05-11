import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import load_dino_model
from src.utils import set_random_seed, plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from src.logger import Logger

import sys
import os.path as osp
import getpass

from datetime import datetime

lr = 1e-3
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
    print(f"Learning Rate: {lr}")

    dataset = ImageFolder(root=args.data_dir, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, 224)]))
    classes = dataset.classes
    
    dinov2 = load_dino_model()


    from tqdm import tqdm
    all_embs = []
    all_labels = []
    def embeddings(dataset):
        for i in tqdm(range(len(dataset))):
            with torch.no_grad():
                out = dinov2(dataset[i][0][None])

            all_embs.append(out)
            all_labels.append(dataset[i][1])

    embeddings(dataset)

    #shuffling dataset
    p = torch.randperm(len(all_labels))
    all_embs = torch.concat(all_embs)[p]
    all_labels = torch.tensor(all_labels, dtype = torch.long)[p]

    n = all_embs.shape[0] * 0.8
    train_embs, val_embs = all_embs[:n], all_embs[n:]
    train_labels, val_labels = all_labels[:n], all_labels[n:]
    
    batch_size = 128

    def get_batch(split = 'train'):
        data_embs = train_embs if split == 'train' else val_embs
        data_labels = train_labels if split == 'train' else val_labels
        idxs = torch.randint(0, len(data_embs), (batch_size, ))
        x = data_embs[idxs]
        y = data_labels[idxs]
        return x.to(device), y.to(device)
    
    @torch.no_grad()
    def estimate_loss(eval_steps):
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_steps)
            for i in range(eval_steps):
                x, y = get_batch()
                logits = model(x)
                losses[i] = F.cross_entropy(logits, y)
            out[split] = losses.mean()
        model.train()
        return out
    
    @torch.no_grad()
    def evaluate(): 
        model.eval() #usless here 
        out = dict()
        out = model(val_embs.to(device)).topk(5).indices
        out['top5'] = (out == val_labels[:, None]).sum() / val_labels.shape[0]# mean was giving error(massive skill issue)
        out['acc'] = (out[:, 0] == val_labels).sum() / val_labels.shape[0]
        model.train()
        return out
            

    model = nn.Sequential(
        nn.Linear(384, 384 * 2),
        nn.ReLU(),
        nn.Linear(384 * 2, len(classes)),
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    writer = SummaryWriter()

    print("\n==========Starting Training========\n")
    max_steps = 5000
    for i in range(max_steps):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            out = estimate_loss()
            print(f"Step {i}/{max_steps}  train_loss = {out['train']:.4f}  val_loss = {out['val']:.4f}")

        losses = estimate_loss(500)
        writer.add_scalar("Loss/train", losses['train'], i)
        writer.add_scalar("Loss/val", losses['val'], i)

    print("\n========Testing on validation set==========\n")
    out = evaluate()
    writer.add_scalar("Accuracy/Top1", out['acc'], i)
    writer.add_scalar("Accuracy/Top5", out['top5'], i)

    torch.save(model.state_dict(), "dino_classi.pth")

if __name__ == "__main__":
    train()
