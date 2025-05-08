import torch.nn as nn
from transformers import AutoModelForImageClassification

def load_vit_model(model="vit_base", num_labels=40):
    if model == "vit_base":
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
        )
    elif model == "swin":
        model = AutoModelForImageClassification.from_pretrained(
            "swin-base-patch4-window7-224"
        )
    elif model == "swin-tiny":
        model = AutoModelForImageClassification.from_pretrained(
            "swin-tiny-patch4-window7-224"
        )
    else:
        raise Exception("Model architecture note supported")

    # remove gradient computation from existing layers
    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, num_labels)

    return model

