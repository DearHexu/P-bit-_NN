"""
Weight visualization/inference example: load a PbitNet checkpoint and view conv layer weight distributions.
Use the same model architecture as during training (e.g. ProbResNet18); checkpoint path should point to outputs/<run_dir>/checkpoint/best.pth.
"""
import os
import argparse
import torch
from matplotlib import pyplot as plt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint/best.pth",
                        help="checkpoint path (e.g. outputs/xxx/checkpoint/best.pth)")
    parser.add_argument("--model", type=str, default="ProbResNet18", choices=["ProbResNet18", "ProbResNet34"])
    args = parser.parse_args()

    if args.model == "ProbResNet18":
        from backbones.ProbResNet import ProbResNet18
        model = ProbResNet18()
    else:
        from backbones.ProbResNet import ProbResNet34
        model = ProbResNet34()

    if not os.path.isfile(args.checkpoint):
        print("Checkpoint not found:", args.checkpoint)
        return
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["net"], strict=True)

    model.eval()
    conv11_weights = model.layer1[0].conv1.weight.data.cpu().numpy().flatten()
    # conv1_weights = model.layer1[0].conv1.weight.data.cpu().numpy()
    conv12_weights = model.layer1[0].conv2.weight.data.cpu().numpy().flatten()
    conv21_weights = model.layer1[1].conv1.weight.data.cpu().numpy().flatten()
    conv22_weights = model.layer1[1].conv2.weight.data.cpu().numpy().flatten()
    # linear_weights = model.linear.weight.data.cpu().numpy().flatten()

    # Plot distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes[0][0].hist(conv11_weights, bins=3, alpha=1, color='red')
    axes[0][0].set_title("Conv11 Layer Weights")

    axes[0][1].hist(conv12_weights, bins=3, alpha=1, color='green')
    axes[0][1].set_title("Conv12 Layer Weights")

    axes[1][0].hist(conv21_weights, bins=3, alpha=1, color='red')
    axes[1][0].set_title("Conv21 Layer Weights")

    axes[1][1].hist(conv22_weights, bins=3, alpha=1, color='green')
    axes[1][1].set_title("Conv22 Layer Weights")
    plt.tight_layout()
    plt.savefig("weight_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved weight_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()
