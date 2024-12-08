import torch
import torch.nn as nn


import os
from torch.utils.data import Dataset

class CustomDirDataset(Dataset):
    def __init__(
        self,
        transcriptions_dir,
        extension=".txt"
    ):
        """
        Args:
            data_dir (str): path to the transcriptions. Format:

                data_dir/
                └── transcriptions
                    ├── UtteranceID1.txt
                    ├── UtteranceID2.txt
                    ...
                    └── UtteranceIDn.txt

            extension (str). By default .txt
        """
        self.transcriptions_dir = transcriptions_dir

        self.utterance_files = [
            f for f in os.listdir(self.transcriptions_dir)
            if f.endswith(extension)
        ]

        if len(self.utterance_files) == 0:
            raise ValueError("No transcription files found in the specified directory.")

    def __len__(self):
        return len(self.utterance_files)

    def __getitem__(self, idx):
        utt_file = self.utterance_files[idx]
        utt_id = os.path.splitext(utt_file)[0]
        with open(os.path.join(self.transcriptions_dir, utt_file), 'r', encoding='utf-8') as f:
            text = f.read().strip()

        return {
            "utt_id": utt_id,
            "text": text
        }
