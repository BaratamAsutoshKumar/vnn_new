import os
import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, out_path):
    """
    Overwrites the loss curve plot every epoch.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train L1", linewidth=2)
    plt.plot(val_losses, label="Val L1", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training Curve (VNN NCLT)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
