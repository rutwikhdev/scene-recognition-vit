import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model.")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/Places2_simp",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="", help="Learning rate scheduler"
    )
    parser.add_argument(
        "--model-name", type=str, default="vit_base", help="Model architecture to use"
    )
    parser.add_argument(
        "--cutmixup", type=bool, default=False, help="Model architecture to use"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="vit_scene_classifier.pth",
        help="Path to save the model checkpoint",
    )

    return parser.parse_args()
