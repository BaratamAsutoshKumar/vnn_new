import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class NCLTVNNDataset(Dataset):
    """
    Robust NCLT dataset loader for VNN training.

    - Never crashes if split files are empty
    - Skips missing h5 / bin files
    - Falls back to all valid sequences automatically
    """

    def __init__(self, root, split="train", num_points=2048):
        self.root = root
        self.num_points = num_points

        # ------------------------
        # Read split file
        # ------------------------
        split_file = osp.join(
            root,
            "train_split.txt" if split == "train" else "valid_split.txt"
        )

        seqs = []
        if osp.isfile(split_file):
            with open(split_file, "r") as f:
                seqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        if len(seqs) == 0:
            print(f"[WARN] Split '{split}' is empty or missing.")
            print("[INFO] Falling back to all available sequences.")
            seqs = [
                d for d in os.listdir(root)
                if osp.isdir(osp.join(root, d)) and d.startswith("201")
            ]

        self.scans = []
        self.rots = []
        self.trans = []

        print(f"[INFO] Loading NCLT split='{split}'")
        print(f"[INFO] Using {len(seqs)} sequences")

        # ------------------------
        # Load sequences
        # ------------------------
        for seq in sorted(seqs):
            seq_dir = osp.join(root, seq)
            h5_path = osp.join(seq_dir, "velodyne_left_False.h5")

            if not osp.isfile(h5_path):
                print(f"[WARN] Missing {h5_path}, skipping sequence {seq}")
                continue

            with h5py.File(h5_path, "r") as f:
                poses = f["poses"][:]               # (T, 12)
                ts = f["valid_timestamps"][:]       # (T,)

            velodyne_dir = osp.join(seq_dir, "velodyne_left")

            for i, t in enumerate(ts):
                bin_path = osp.join(velodyne_dir, f"{int(t)}.bin")
                if not osp.isfile(bin_path):
                    continue

                self.scans.append(bin_path)

                pose = poses[i].reshape(3, 4)
                self.rots.append(pose[:, :3])
                self.trans.append(pose[:, 3])

        # ------------------------
        # Final report (NO ASSERT)
        # ------------------------
        if len(self.scans) == 0:
            print("[ERROR] No valid NCLT scans found!")
            print("[ERROR] Check dataset path and h5 files.")
        else:
            print(
                f"[INFO] Loaded {len(self.scans)} samples "
                f"from {len(seqs)} sequences"
            )

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = np.fromfile(self.scans[idx], dtype=np.float32)
        scan = scan.reshape(-1, 4)[:, :3]      # (N,3)

        R = self.rots[idx]
        t = self.trans[idx]

        scan_gt = (R @ scan.T).T + t

        scan = torch.from_numpy(scan).float().T      # (3,N)
        scan_gt = torch.from_numpy(scan_gt).float().T

        return scan, scan_gt
