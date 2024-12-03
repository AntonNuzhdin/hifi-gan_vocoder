import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.utils.melspec import MelSpectrogram, MelSpectrogramConfig


class HiFiGANLoss(nn.Module):
    def __init__(
        self,
        lambda_fm: float = 2,
        lambda_mel: float = 45
    ):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.mel_spectrogram = MelSpectrogram(MelSpectrogramConfig)

    def discriminator_loss(self, outs_predicted, outs_gt, **batch):
        disc_loss = 0.0
        for pred_out, gt_out in zip(outs_predicted, outs_gt):
            disc_loss += torch.mean((gt_out - 1) ** 2) + torch.mean(pred_out ** 2)

        return {
            "disc_loss": disc_loss
        }

    def generator_loss(self, mel_gt, prediction, outs_predicted, fmaps_predicted, fmaps_gt, **batch):
        # Adv loss
        loss_adv = sum(torch.mean((pred_out - 1) ** 2) for pred_out in outs_predicted)

        # Mel spec loss
        predicted_mel = self.mel_spectrogram(prediction).squeeze(1)
        loss_mel = self.lambda_mel * F.l1_loss(predicted_mel, mel_gt)

        # Feature matching loss
        loss_fm = 0.0
        for fmap_pred, fmap_gt in zip(fmaps_predicted, fmaps_gt):
            loss_fm += self.lambda_fm * F.l1_loss(fmap_pred, fmap_gt)

        total_loss = loss_adv + loss_mel + loss_fm

        return {
            "gen_loss": total_loss,
            "loss_adv": loss_adv,
            "loss_mel": loss_mel,
            "loss_fm": loss_fm,
        }
