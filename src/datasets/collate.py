import torch
import torch.nn.functional as F

from src.utils.melspec import MelSpectrogram, MelSpectrogramConfig

import random


def collate_fn(batch):
    get_mel = MelSpectrogram(MelSpectrogramConfig())

    wav_gt = []
    mel_gt = []

    for item in batch:
        audio = item["audio"].squeeze(0)
        audio_len = audio.size(0)

        if audio_len < 8192:
            pad_size = 8192 - audio_len
            audio = F.pad(audio, (0, pad_size), value=0)
        else:
            start_idx = random.randint(0, audio_len - 8192)
            audio = audio[start_idx:start_idx + 8192]

        wav_gt.append(audio)
        mel_gt.append(get_mel(audio.unsqueeze(0)))

    output = {
        "wav_gt": pad_1D(wav_gt).unsqueeze(1),
        "mel_gt": pad_2D(mel_gt)
    }
    return output


def pad_1D(inputs, pad_value=0):
    max_len = max(x.size(0) for x in inputs)
    padded = [F.pad(x, (0, max_len - x.size(0)), value=pad_value) for x in inputs]
    return torch.stack(padded)


def pad_2D(inputs, pad_value=0, maxlen=None):
    if maxlen is None:
        maxlen = max(x.size(-1) for x in inputs)
    padded = [F.pad(x, (0, maxlen - x.size(-1)), value=MelSpectrogramConfig().pad_value) for x in inputs]
    return torch.stack(padded)


# def collate_fn(batch):
#     get_mel = MelSpectrogram(MelSpectrogramConfig())

#     wav_gt = [item["audio"].squeeze(0) for item in batch]
#     print(len(wav_gt[0]))
#     mel_gt = [get_mel(item["audio"]) for item in batch]
#     output = {
#         "wav_gt": pad_1D(wav_gt).unsqueeze(1),
#         "mel_gt": pad_2D(mel_gt)
#     }
#     return output
