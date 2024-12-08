import os
from scipy.io.wavfile import write
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from wvmos import get_wvmos

from src.datasets.custom_dataset import CustomDirDataset
from src.model import HIFIGAN


tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')



@hydra.main(config_path="src/configs", config_name="custom_dir")
def main(cfg: DictConfig):
    hifigan = HIFIGAN()
    state_dict = torch.load(cfg.model.hifigan_checkpoint, map_location='cuda:0')
    hifigan.load_state_dict(state_dict['state_dict'])
    hifigan.eval()
    if torch.cuda.is_available():
        hifigan = hifigan.cuda()

    print(f'The predictions will be stored in {cfg.model.output_dir}')
    os.makedirs(cfg.model.output_dir, exist_ok=True)

    model = get_wvmos(cuda=True)

    def text_to_audio(text, output_path):
        sequences, lengths = utils.prepare_input_sequence([text])
        sequences = sequences.to('cuda')
        lengths = lengths.to('cuda')

        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences, lengths)  # mel: (B, n_mels, T)

        mel = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        with torch.no_grad():
            output = hifigan(mel)
            audio = output["prediction"].cpu().numpy().squeeze(0)

        audio = ((audio + 1) * 127.5).astype(np.uint8)

        write(output_path, 22040, audio[0])
        print(f"Saved generated audio to {output_path}")

        mos_score = model.calculate_one(output_path)
        print(f"WV-MOS score for {output_path}: {mos_score}")

    if cfg.text is not None:
        output_path = os.path.join(cfg.model.output_dir, "generated_from_text.wav")
        text_to_audio(cfg.text, output_path)
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
                output_path = os.path.join(cfg.model.output_dir, f"{utt_id}.wav")
                text_to_audio(text, output_path)

if __name__ == "__main__":
    main()
