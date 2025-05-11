import torch.nn as nn
from transformers import AutoModelForImageClassification

def load_vit_model(model="vit_base", num_labels=40):
    if model == "vit_base":
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
        )
    elif model == "swin":
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
    elif model == "swinv2":
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256"
        )
    elif model == "dinov2":
        model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base"
        )
    else:
        raise Exception("Model architecture not supported")

    # remove gradient computation from existing layers
    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, num_labels)

    return model

