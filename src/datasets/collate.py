import torch
import torch.nn.functional as F
import random

from src.utils.melspec import MelSpectrogram, MelSpectrogramConfig

mel_config = MelSpectrogramConfig()

def collate_fn(batch):
    get_mel = MelSpectrogram(mel_config)

    wav_gt = []
    mel_gt = []

    for item in batch:
        if audio.dim() == 2 and audio.size(0) == 1:
            audio = audio.squeeze(0)

        audio_len = audio.size(0)
        if audio_len < 8192:
            pad_size = 8192 - audio_len
            audio = F.pad(audio, (0, pad_size), value=0.0)
        else:
            start_idx = random.randint(0, audio_len - 8192)
            audio = audio[start_idx:start_idx + 8192]

        wav_gt.append(audio)

        mel = get_mel(audio.unsqueeze(0).unsqueeze(0))  # (1, n_mels, time)
        mel = mel.squeeze(0)
        mel_gt.append(mel)

    wav_gt = pad_1D(wav_gt).unsqueeze(1)

    mel_gt = pad_2D(mel_gt, pad_value=mel_config.pad_value)

    output = {
        "wav_gt": wav_gt,
        "mel_gt": mel_gt
    }

    return output


def pad_1D(inputs, pad_value=0.0):
    max_len = max(x.size(0) for x in inputs)
    padded = [F.pad(x, (0, max_len - x.size(0)), value=pad_value) for x in inputs]
    return torch.stack(padded)


def pad_2D(inputs, pad_value=None, maxlen=None):
    if pad_value is None:
        pad_value = mel_config.pad_value

    if maxlen is None:
        maxlen = max(x.size(-1) for x in inputs)
    padded = [F.pad(x, (0, maxlen - x.size(-1)), value=pad_value) for x in inputs]
    return torch.stack(padded)
