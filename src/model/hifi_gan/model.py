import torch.nn as nn

from src.model.hifi_gan.generator import Generator
from src.model.hifi_gan.mpd import MPD
from src.model.hifi_gan.msd import MSD


#  hu   ku             kr             Dr
# V1 512 [16, 16, 4, 4] [3, 7, 11] [[1, 1], [3, 1], [5, 1]] ×3

class HIFIGAN(nn.Module):
    def __init__(
        self,
        input_channels: int = 80,
        hu: int = 512,
        ku: list[int] = [16, 16, 4, 4],
        kr: list[int] = [3, 7, 11],
        Dr: list[list[list[int]]] = [
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
        ]
    ):
        super().__init__()

        self.generator = Generator(input_channels, hu, ku, kr, Dr)
        self.MPD = MPD()
        self.MSD = MSD()

    def forward(self, mel_gt, **batch):
        return {
            "prediction": self.generator(mel_gt.squeeze(1))
        }

    def discriminator_forward(self, prediction, wav_gt, **batch):
        outs_pred = []
        fmaps_pred = []
        outs_gt = []
        fmaps_gt = []

        # Для предсказаний
        mpd_out_pred, mpd_fmap_pred = self.MPD(prediction)
        msd_out_pred, msd_fmap_pred = self.MSD(prediction)

        outs_pred.extend(mpd_out_pred)
        outs_pred.extend(msd_out_pred)
        fmaps_pred.extend(mpd_fmap_pred)
        fmaps_pred.extend(msd_fmap_pred)

        # Для ground truth
        mpd_out_gt, mpd_fmap_gt = self.MPD(wav_gt)
        msd_out_gt, msd_fmap_gt = self.MSD(wav_gt)

        outs_gt.extend(mpd_out_gt)
        outs_gt.extend(msd_out_gt)
        fmaps_gt.extend(mpd_fmap_gt)
        fmaps_gt.extend(msd_fmap_gt)

        return {
            "outs_predicted": outs_pred,
            "fmaps_predicted": fmaps_pred,
            "outs_gt": outs_gt,
            "fmaps_gt": fmaps_gt,
        }

    def __str__(self):
        return (
            f"Generator:\n{self.__get_model_params(self.generator)}\n"
            f"MPD:\n{self.__get_model_params(self.MPD)}\n"
            f"MSD:\n{self.__get_model_params(self.MSD)}\n"
        )

    def __get_model_params(self, model_):
        all_parameters = sum([p.numel() for p in model_.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in model_.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = f"All parameters: {all_parameters}\n"
        result_info += f"Trainable parameters: {trainable_parameters}"

        return result_info
