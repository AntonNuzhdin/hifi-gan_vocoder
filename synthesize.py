import os
import argparse
import torch
import soundfile as sf
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from src.datasets.custom_dataset import CustomDirDataset


tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

from src.model import HIFIGAN


@hydra.main(config_path="src/configs", config_name="custom_dir")
def main(cfg: DictConfig):
    hifigan = HIFIGAN()
    state_dict = torch.load(cfg.model.hifigan_checkpoint, map_location='cuda:0')
    hifigan.load_state_dict(state_dict['state_dict'])
    hifigan.eval()
    if torch.cuda.is_available():
        hifigan = hifigan.cuda()

    os.makedirs(cfg.model.output_dir, exist_ok=True)

    def text_to_audio(text):
        sequences, lengths = utils.prepare_input_sequence([text])
        sequences = sequences.to('cuda')
        lengths = lengths.to('cuda')

        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences, lengths)  # mel: (B, n_mels, T)

        mel = mel.unsqueeze(1)  # (B, 1, n_mels, T)

        with torch.no_grad():
            output = hifigan(mel)
            audio = output["prediction"].cpu().numpy().squeeze(0)

        return audio

    if cfg.text is not None:
        audio = text_to_audio(args.text)
        output_path = os.path.join(cfg.model.output_dir, "generated_from_text.wav")
        sf.write(output_path, audio, 22050)
        print(f"Saved generated audio to {output_path}")
    else:
        dataset = CustomDirDataset(
            data_dir=cfg.data.data_dir,
            extension=cfg.data.extension
        )
        dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size, num_workers=cfg.inference.num_workers)

        for batch_data in dataloader:
            utt_ids = batch_data["utt_id"]
            texts = batch_data["text"]
            for utt_id, text in zip(utt_ids, texts):
                audio = text_to_audio(text)
                output_path = os.path.join(cfg.model.output_dir, f"{utt_id}.wav")
                sf.write(output_path, audio, 22050)
                print(f"Saved generated audio: {output_path}")

if __name__ == "__main__":
    main()
