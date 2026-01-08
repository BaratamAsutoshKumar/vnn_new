import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def visualize_and_save(scan, gt, pred, out_path, title=""):
    """
    scan, gt, pred: tensors of shape (3, N)
    out_path: full path to save image, e.g. "vis/epoch_10_sample_0.png"
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    scan = scan.detach().cpu().numpy().T
    gt   = gt.detach().cpu().numpy().T
    pred = pred.detach().cpu().numpy().T

    fig = plt.figure(figsize=(12, 4))

    for i, (pts, name) in enumerate([
        (scan, "Input"),
        (gt, "GT"),
        (pred, "Pred")
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(name)
        ax.axis("off")

        # equal aspect ratio (important for geometry)
        max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
        mid = pts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
