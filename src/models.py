import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


def load_vit_model(model_name="vit_base", num_labels=40):
    if model_name == "vit_base":
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
        )
    elif model_name == "swin":
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
    elif model_name == "swinv2":
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256"
        )
    elif model_name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    else:
        raise Exception("Model architecture not supported")

    # remove gradient computation from existing layers
    if model_name != 'dinov2':
        for p in model.parameters():
            p.requires_grad = False

        model.classifier = nn.Linear(model.classifier.in_features, num_labels)

    return model
