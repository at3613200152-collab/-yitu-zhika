import os
import yaml
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def tensor_to_image(tensor, normalize=True):
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))
    if normalize:
        img = (img * 0.5 + 0.5) * 255.0
    else:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_image(tensor, path, normalize=True):
    img = tensor_to_image(tensor, normalize)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


def plot_comparison(rgb, real_nir, pred_nir, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    rgb_img = tensor_to_image(rgb)
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")
    
    real_nir_img = tensor_to_image(real_nir)
    if real_nir_img.shape[2] == 1:
        real_nir_img = real_nir_img[:, :, 0]
    axes[1].imshow(real_nir_img, cmap="gray")
    axes[1].set_title("Real NIR")
    axes[1].axis("off")
    
    pred_nir_img = tensor_to_image(pred_nir)
    if pred_nir_img.shape[2] == 1:
        pred_nir_img = pred_nir_img[:, :, 0]
    axes[2].imshow(pred_nir_img, cmap="gray")
    axes[2].set_title("Predicted NIR")
    axes[2].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
