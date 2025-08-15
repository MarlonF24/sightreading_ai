from torch import LongTensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import *
from tokeniser.tokeniser import MyTokeniser
import json, constants as constants

class MyTokenDataset(Dataset):
    def __init__(self, files_paths: Sequence[Path], tokeniser: MyTokeniser, max_sequence_length: int, bos_token_id: int, eos_token_id: int, pad_token_id: int, sort_by_length: bool = True):
        """Dataset for tokenised files from a MyTokeniser.

        :param files_paths: Paths to the JSON files containing tokenised data.
        :type files_paths: Sequence[Path]
        :param tokeniser: Instance of MyTokeniser used for encoding the data.
        :type tokeniser: MyTokeniser
        :param max_sequence_length: Maximum sequence length for the input IDs.
        :type max_sequence_length: int
        :param bos_token_id: Beginning of sequence token ID.
        :type bos_token_id: int
        :param eos_token_id: End of sequence token ID.
        :type eos_token_id: int
        :param pad_token_id: Padding token ID.
        :type pad_token_id: int
        :param sort_by_length: Whether to sort the dataset by sequence length.
        :type sort_by_length: bool
        :raises TypeError: If the tokeniser is not an instance of MyTokeniser.
        :raises ValueError: If no valid files are found.
        """
        if not isinstance(tokeniser, MyTokeniser):
            raise TypeError(f"Expected tokeniser to be MyTokeniser (thats what this dataloader is designed for), got {type(tokeniser)}")

        self.tokeniser = tokeniser 
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.files_paths = [
            file for file in files_paths
            if (temp := json.loads(file.read_text()))[constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY] == self.tokeniser.hexa_hash and
               len(temp[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY]) <= max_sequence_length
        ]

        if not self.files_paths:
            raise ValueError(f"No files found with matching given tokeniser hash {self.tokeniser.hexa_hash}. "
                             f"Retokenise the data with your given tokeniser.")


        print(f"Filtered dataset size: {len(self.files_paths)} files (from given {len(files_paths)}) with matching tokeniser hash {self.tokeniser.hexa_hash} and length <= {max_sequence_length} (see 'max_position_embeddings' specified in model config)")


        if sort_by_length:
            # Sort files by length of input_ids so that when sequences in a batch have minimal length difference and thus minimal padding 
            self.files_paths.sort(key=lambda file: len(json.loads(file.read_text())[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY]))

            print(f"Dataset sorted by sequence length")
            
            print(f"Length range: {len(json.loads(self.files_paths[0].read_text())[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY])} to "
                  f"{len(json.loads(self.files_paths[-1].read_text())[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY])}")

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        with self.files_paths[idx].open("r") as f:
            data = json.load(f)
            input_ids = data[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY]
            input_ids.insert(0, self.bos_token_id)
            input_ids.append(self.eos_token_id)
            labels = data[constants.tokeniser_constants.TOKENS_LABELS_KEY]


            if len(input_ids) < len(labels):
                raise ValueError(f"Corrupt data. Found Input IDs length ({len(input_ids)}) is less than labels length ({len(labels)}) in file {self.files_paths[idx]}. Before training, ensure that the data is correctly tokenised with a MyTokeniser.")

            # Pad the labels so that metadata and bos/eos tokens are not predicted
            labels = [-100] * (len(input_ids) - len(labels) - 1) + data[constants.tokeniser_constants.TOKENS_LABELS_KEY] + [-100]

            res = {
                constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY: LongTensor(input_ids),
                constants.tokeniser_constants.TOKENS_LABELS_KEY: LongTensor(labels),
            }


            # print(f"Processed file: {res}")
            return res
            

                
