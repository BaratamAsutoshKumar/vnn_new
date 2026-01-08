import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm

from models.vn_pointnet_regression import VNPointNetRegressor
from data_utils.nclt_vnn_dataset import NCLTVNNDataset
from viz_pointcloud import visualize_and_save
from utils.loss_plot import plot_loss
from utils.checkpoint import save_checkpoint, load_checkpoint


# =======================
# Config
# =======================
ROOT = "/DATA/common/NCLT"
NUM_POINTS = 130000          # using all points (heavy!)
BATCH_SIZE = 1
EPOCHS = 500
LR = 1e-3
NUM_WORKERS = 0
VIS_EVERY = 10

PATIENCE = 25
MIN_DELTA = 1e-4

CKPT_DIR = "/DATA2/asutosh/vnn_data/models/checkpoints"
VIS_DIR = "/DATA2/asutosh/vnn_data/visualizations"
LOSS_PLOT_PATH = os.path.join(VIS_DIR, "loss_curve.png")

RESUME = True
RESUME_PATH = os.path.join(CKPT_DIR, "last.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =======================
# Reproducibility
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =======================
# Epoch runner
# =======================
def run_epoch(model, loader, optimizer=None, scaler=None, epoch=0, split="train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0

    # tqdm over batches
    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"{split.upper()} Epoch {epoch}",
        leave=False
    )

    for it, (scan, scan_gt) in pbar:
        scan = scan.unsqueeze(0).to(DEVICE) 
        scan_gt = scan_gt.unsqueeze(0).to(DEVICE)

        # (B,3,N) -> (B,1,3,N)
        scan_vnn = scan.unsqueeze(1) 

        with amp.autocast("cuda", enabled=is_train):
            pred = model(scan_vnn)     # (B,1,3,N)
            pred = pred.squeeze(1)     # (B,3,N)
            loss = F.l1_loss(pred, scan_gt)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_val = loss.item()
        total_loss += loss_val

        # ---- tqdm live update ----
        pbar.set_postfix(
            loss=f"{loss_val:.6f}",
            avg=f"{total_loss / (it + 1):.6f}"
        )

        # ---- optional per-step print (useful for nohup logs) ----
        if it % 1000 == 0:
            print(
                f"[{split}] Epoch {epoch:03d} | "
                f"Step {it:05d}/{len(loader)} | "
                f"Loss {loss_val:.6f}",
                flush=True
            )

        # ---- visualization (overwrite) ----
        if split == "train" and it == 0 and epoch % VIS_EVERY == 0:
            visualize_and_save(
                scan[0],
                scan_gt[0],
                pred[0],
                out_path=os.path.join(VIS_DIR, "train_vis.png"),
                title=f"Train | Epoch {epoch}"
            )

    return total_loss / len(loader)


# =======================
# Main
# =======================

def collate_single(batch):
    # batch is a list of length 1
    return batch[0]

def train():
    set_seed(42)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # -------- Dataset --------
    train_dataset = NCLTVNNDataset(ROOT, split="train", num_points=NUM_POINTS)
    val_dataset   = NCLTVNNDataset(ROOT, split="valid", num_points=NUM_POINTS)

    use_val = len(val_dataset) > 0
    if not use_val:
        print("[WARN] Validation dataset empty. Skipping validation.", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,          # IMPORTANT
        collate_fn=collate_single,
        pin_memory=False
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_single,
        pin_memory=False
    )

    # -------- Model --------
    model = VNPointNetRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = amp.GradScaler("cuda")

    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val = float("inf")
    epochs_since_improve = 0

    # -------- Resume --------
    if RESUME and os.path.exists(RESUME_PATH):
        print(f"Resuming from {RESUME_PATH}", flush=True)
        start_epoch, train_losses, val_losses, best_val, epochs_since_improve = load_checkpoint(
            RESUME_PATH, model, optimizer, scaler
        )
        start_epoch += 1

    # -------- Epoch loop --------
    epoch_pbar = tqdm(
        range(start_epoch, EPOCHS),
        desc="TRAINING",
        position=0
    )

    for epoch in epoch_pbar:
        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, epoch, split="train"
        )

        if use_val:
            val_loss = run_epoch(
                model, val_loader, optimizer=None, scaler=None, epoch=epoch, split="val"
            )
        else:
            val_loss = train_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---- epoch summary ----
        print(
            f"\nEpoch {epoch:03d} | "
            f"Train L1: {train_loss:.6f} | "
            f"Val L1: {val_loss:.6f}",
            flush=True
        )

        epoch_pbar.set_postfix(
            train=f"{train_loss:.6f}",
            val=f"{val_loss:.6f}"
        )

        # ---- loss curve (overwrite) ----
        if use_val:
            plot_loss(train_losses, val_losses, LOSS_PLOT_PATH)
        else:
            plot_loss(train_losses, train_losses, LOSS_PLOT_PATH)

        # ---- early stopping ----
        if best_val - val_loss > MIN_DELTA:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pth"))
        else:
            epochs_since_improve += 1

        print(
            f"EarlyStop counter: {epochs_since_improve}/{PATIENCE}",
            flush=True
        )

        # ---- save last ----
        save_checkpoint(
            os.path.join(CKPT_DIR, "last.pth"),
            model,
            optimizer,
            scaler,
            epoch,
            train_losses,
            val_losses,
            best_val,
            epochs_since_improve
        )

        # ---- periodic snapshot ----
        if epoch % VIS_EVERY == 0:
            torch.save(
                model.state_dict(),
                os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")
            )

        if epochs_since_improve >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break


if __name__ == "__main__":
    train()
