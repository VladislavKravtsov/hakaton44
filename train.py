r"""
Trains the 4-class ore segmentation model (background/ordinary/thin/talc) on
the combined output of lumenstone_pipeline.py and build_own_dataset.py.

Tuned for a GTX 1660 Super (6GB VRAM, no tensor cores -- plain fp32, no AMP,
which was both unstable (NaNs under fp16 autocast with this encoder) and not
actually faster on a non-tensor-core GPU).

Usage:
    python train.py ^
        --data-root ..\lumenstone_pipeline\lumenstone_prepared ^
        --data-root own_dataset_prepared ^
        --epochs 40 --batch-size 8

Multiple --data-root flags are merged into one training set. Each root must
contain train/images, train/masks, val/images, val/masks (exactly what both
prep pipelines already produce).

Checkpoints go to .\checkpoints\best_model.pt (loaded by inference.py once
you wire it in -- see predict_class_mask()).
"""
import argparse
import os
import time

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

NUM_CLASSES = 4  # background, ordinary, thin, talc
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_file_list(roots: list, split: str) -> list:
    files = []
    for root in roots:
        img_dir = os.path.join(root, split, "images")
        mask_dir = os.path.join(root, split, "masks")
        if not os.path.isdir(img_dir):
            print(f"  (skip) {img_dir} does not exist")
            continue
        for name in sorted(os.listdir(img_dir)):
            mp = os.path.join(mask_dir, name)
            if os.path.isfile(mp):
                files.append((os.path.join(img_dir, name), mp))
    return files


def get_transforms(tile_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.PadIfNeeded(min_height=tile_size, min_width=tile_size, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(height=tile_size, width=tile_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    return A.Compose([
        A.PadIfNeeded(min_height=tile_size, min_width=tile_size, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(height=tile_size, width=tile_size),
    ])


class OreDataset(Dataset):
    def __init__(self, files: list, tile_size: int, train: bool, repeat: int = 1):
        """repeat > 1 makes one epoch draw that many random crops from every
        source image. The photos are ~2500x3400 while a crop is 512x512, so a
        single crop per image per epoch would leave >95% of each image unseen
        between checkpoints; repeating keeps the DataLoader simple and the
        epoch a meaningful unit."""
        self.files = files
        self.repeat = max(1, repeat) if train else 1
        self.transform = get_transforms(tile_size, train)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        img_path, mask_path = self.files[idx % len(self.files)]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]

        augmented = self.transform(image=img, mask=mask)
        img, mask = augmented["image"], augmented["mask"]

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion_matrix_update(cm: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    k = (target >= 0) & (target < NUM_CLASSES)
    cm += np.bincount(
        NUM_CLASSES * target[k].astype(int) + pred[k].astype(int), minlength=NUM_CLASSES ** 2
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-9), 0.0)


def compute_class_weights(files: list, sample_size: int = 60) -> torch.Tensor:
    """Inverse-frequency CE weights from a sample of training masks.
    Thin intergrowths and talc are a few percent of all pixels; with
    unweighted CE the model scores ~0 F1 on them (confirmed empirically on
    this dataset) because predicting background/ordinary everywhere is
    almost as cheap."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(files), min(sample_size, len(files)), replace=False)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in idx:
        m = cv2.imread(files[i][1], cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        vals, c = np.unique(m, return_counts=True)
        for v, cnt in zip(vals, c):
            if 0 <= v < NUM_CLASSES:
                counts[v] += cnt
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (NUM_CLASSES * counts)
    # cap so one ultra-rare class can't dominate the loss with 100x weight
    weights = np.minimum(weights, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion_dice, criterion_ce, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    torch.set_grad_enabled(train)
    pbar = tqdm(loader, desc="train" if train else "val  ", ncols=100)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)

        if train:
            optimizer.zero_grad()
        pred = model(img)
        loss = criterion_dice(pred, mask) + criterion_ce(pred, mask)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * img.size(0)
        pred_labels = pred.argmax(dim=1).detach().cpu().numpy()
        confusion_matrix_update(cm, pred_labels, mask.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_per_class(cm)
    return avg_loss, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", action="append", required=True, help="repeatable: one or more prepared dataset roots")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6, help="early stop after N epochs with no val F1 improvement")
    parser.add_argument("--repeat", type=int, default=6, help="random crops drawn per source image per epoch")
    parser.add_argument("--arch", default="unet", choices=["unet", "deeplabv3plus"],
                        help="segmentation architecture (both use a resnet34 ImageNet encoder)")
    parser.add_argument("--init-from", default=None, metavar="CKPT",
                        help="warm-start: load model weights from this checkpoint before training "
                             "(arch must match; optimizer state starts fresh)")
    parser.add_argument("--gpu-mem-fraction", type=float, default=None, metavar="F",
                        help="cap this process's VRAM at F of total (e.g. 0.5 = 3GB on a 6GB card); "
                             "lower --batch-size accordingly or expect OOM")
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found, training will be very slow.")
    if args.gpu_mem_fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM cap: {args.gpu_mem_fraction:.0%} of {total_gb:.1f}GB = {args.gpu_mem_fraction * total_gb:.1f}GB")

    train_files = build_file_list(args.data_root, "train")
    val_files = build_file_list(args.data_root, "val")
    print(f"train tiles source images: {len(train_files)}, val: {len(val_files)}")
    if not train_files or not val_files:
        raise SystemExit("No train/val files found -- check --data-root paths.")

    train_ds = OreDataset(train_files, args.tile_size, train=True, repeat=args.repeat)
    val_ds = OreDataset(val_files, args.tile_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    arch_cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[args.arch]
    model = arch_cls(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(device)
    print(f"Architecture: {args.arch} (resnet34 encoder)")
    warm_start_f1 = -1.0
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        ckpt_arch = ckpt.get("arch", "unet")
        if ckpt_arch != args.arch:
            raise SystemExit(f"--init-from checkpoint is '{ckpt_arch}' but --arch is '{args.arch}'")
        model.load_state_dict(ckpt["model_state"])
        # inherit the loaded checkpoint's score so a post-restart dip epoch
        # can't overwrite a better checkpoint with a worse one
        warm_start_f1 = float(ckpt.get("val_fg_f1", -1.0))
        print(f"Warm start from {args.init_from} (epoch {ckpt.get('epoch')}, val_fg_f1={warm_start_f1:.3f})")
    print("Estimating class weights from training masks...")
    class_weights = compute_class_weights(train_files).to(device)
    print(f"  class weights: bg={class_weights[0]:.2f} ordinary={class_weights[1]:.2f} "
          f"thin={class_weights[2]:.2f} talc={class_weights[3]:.2f}")
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")
    criterion_ce = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(args.output, exist_ok=True)
    class_names = ["background", "ordinary", "thin", "talc"]

    best_f1 = warm_start_f1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_f1 = run_epoch(model, train_loader, criterion_dice, criterion_ce, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion_dice, criterion_ce, optimizer, device, train=False)
        fg_f1 = val_f1[1:].mean()  # ordinary/thin/talc, excluding background
        scheduler.step(fg_f1)

        dt = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs} ({dt:.0f}s) "
              f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} fg_f1={fg_f1:.3f}")
        for name, f1 in zip(class_names, val_f1):
            print(f"    {name:10s} F1={f1:.3f}")

        if fg_f1 > best_f1:
            best_f1 = fg_f1
            epochs_without_improvement = 0
            ckpt_path = os.path.join(args.output, "best_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "encoder_name": "resnet34",
                "num_classes": NUM_CLASSES,
                "epoch": epoch,
                "val_fg_f1": fg_f1,
                "imagenet_mean": IMAGENET_MEAN.tolist(),
                "imagenet_std": IMAGENET_STD.tolist(),
            }, ckpt_path)
            print(f"    -> saved new best checkpoint ({ckpt_path}), fg_f1={fg_f1:.3f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping early.")
                break

    print(f"\nDone. Best val fg_f1={best_f1:.3f}. Checkpoint: {os.path.join(args.output, 'best_model.pt')}")


if __name__ == "__main__":
    main()
