from torch import LongTensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import *
from tokeniser.tokeniser import MyTokeniser
import json, constants as constants

class MyTokenDataset(Dataset):
    def __init__(self, files_paths: Sequence[Path], tokeniser: MyTokeniser, bos_token_id: int, eos_token_id: int, pad_token_id: int):
        """
        Dataset for tokenised files from a MyTokeniser.

        :param bos_token_id: Beginning of sequence token ID.
        :param eos_token_id: End of sequence token ID.
        :param pad_token_id: Padding token ID.
        :param files_paths: Paths to the JSON files containing tokenised data.
        :type files_paths: Sequence[Path]
        :param tokeniser_hash: Hash of the tokeniser used for encoding the data.
        :type tokeniser_hash: str
        """
        if not isinstance(tokeniser, MyTokeniser):
            raise TypeError(f"Expected tokeniser to be MyTokeniser (thats what this dataloader is designed for), got {type(tokeniser)}")

        self.tokeniser = tokeniser 
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.files_paths = [
            file for file in files_paths
            if json.loads(file.read_text())[constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY] == self.tokeniser.hexa_hash
        ]

        if not self.files_paths:
            raise ValueError(f"No files found with matching given tokeniser hash {self.tokeniser.hexa_hash}. "
                             f"Retokenise the data with your given tokeniser.")


        print(f"Filtered dataset size: {len(self.files_paths)} files (from given {len(files_paths)}) with matching tokeniser hash {self.tokeniser.hexa_hash}")

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        with self.files_paths[idx].open("r") as f:
            data = json.load(f)
            input_ids = data[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY]
            input_ids.insert(0, self.bos_token_id)
            input_ids.append(self.eos_token_id)

            if len(input_ids) < len(data[constants.tokeniser_constants.TOKENS_LABELS_KEY]):
                raise ValueError(f"Corrupt data. Found Input IDs length ({len(input_ids)}) is less than labels length ({len(data[constants.tokeniser_constants.TOKENS_LABELS_KEY])}) in file {self.files_paths[idx]}. Before training, ensure that the data is correctly tokenised with a MyTokeniser.")

            labels = [self.pad_token_id] * (len(input_ids) - len(labels)) + data[constants.tokeniser_constants.TOKENS_LABELS_KEY]

            return {
                constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY: LongTensor(input_ids),
                constants.tokeniser_constants.TOKENS_LABELS_KEY: LongTensor(labels),
            }

                
