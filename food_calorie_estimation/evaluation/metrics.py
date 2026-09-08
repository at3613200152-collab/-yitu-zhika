import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def calculate_psnr(pred, target, data_range=1.0):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    if pred.ndim == 4:
        pred = pred[0]
        target = target[0]
    
    pred = np.transpose(pred, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))
    
    if pred.shape[2] == 1:
        pred = pred[:, :, 0]
        target = target[:, :, 0]
    
    psnr = peak_signal_noise_ratio(target, pred, data_range=data_range)
    return psnr


def calculate_ssim(pred, target, data_range=1.0):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    if pred.ndim == 4:
        pred = pred[0]
        target = target[0]
    
    pred = np.transpose(pred, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))
    
    if pred.shape[2] == 1:
        pred = pred[:, :, 0]
        target = target[:, :, 0]
    
    ssim = structural_similarity(target, pred, data_range=data_range, channel_axis=-1 if pred.ndim == 3 else None)
    return ssim


def calculate_mape(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    target = np.maximum(target, 1e-8)
    mape = np.mean(np.abs((target - pred) / target)) * 100
    return mape


def calculate_rmse(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    return rmse


def calculate_accuracy(pred_logits, target):
    if isinstance(pred_logits, torch.Tensor):
        pred_classes = torch.argmax(pred_logits, dim=1)
        correct = (pred_classes == target).sum().item()
        total = target.size(0)
    else:
        pred_classes = np.argmax(pred_logits, axis=1)
        correct = (pred_classes == target).sum()
        total = len(target)
    
    return correct / total * 100
