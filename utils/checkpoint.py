import torch

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    train_losses,
    val_losses,
    best_val,
    epochs_since_improve
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "epochs_since_improve": epochs_since_improve,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    epoch = ckpt.get("epoch", 0)
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])
    best_val = ckpt.get("best_val", float("inf"))
    epochs_since_improve = ckpt.get("epochs_since_improve", 0)

    return epoch, train_losses, val_losses, best_val, epochs_since_improve
