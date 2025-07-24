from torch import LongTensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import *
import json

class MyTokenDataset(Dataset):
    def __init__(self, files_paths: Sequence[Path], tokeniser_hash: str, bos_token_id: int, eos_token_id: int, pad_token_id: int):
        """Dataset for tokenised files with filtering by tokeniser hash.

        :param files_paths: Paths to the JSON files containing tokenised data.
        :type files_paths: Sequence[Path]
        :param tokeniser_hash: Hash of the tokeniser used for encoding the data.
        :type tokeniser_hash: str
        """
        self.tokeniser_hash = tokeniser_hash
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.files_paths = [
            file for file in files_paths
            if json.loads(file.read_text())["tokeniser_hash"] ==self.tokeniser_hash
        ]
        if not self.files_paths:
            raise ValueError(f"No files found with tokeniser hash {self.tokeniser_hash}. "
                             f"Check the tokeniser used for encoding the data.")
        
        print(f"Filtered dataset size: {len(self.files_paths)} files (from {len(files_paths)}) with tokeniser hash {self.tokeniser_hash}")
    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        with self.files_paths[idx].open("r") as f:
            data = json.load(f)
            input_ids = data["input_ids"]
            input_ids.insert(0, self.bos_token_id)
            input_ids.append(self.eos_token_id)

            labels = self.prepadding(input_ids, data["labels"])

        return {
            "input_ids": LongTensor(input_ids),
            "labels": LongTensor(labels),
        }
    
    def prepadding(self, input_ids: list, labels: list) -> list:
        """Pre-padding the labels on the left."""
        if (len(input_ids) - len(labels)) > 0:
            return [self.pad_token_id] * (len(input_ids) - len(labels)) + labels

        return  labels
