from torch import LongTensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import *
from tokeniser.tokeniser import MyTokeniser
import json, constants as constants

class MyTokenDataset(Dataset):
    """
    PyTorch Dataset for loading and processing tokenized music files.
    
    This dataset handles JSON files containing tokenized musical sequences from MyTokeniser,
    providing proper sequence formatting with BOS/EOS tokens and label alignment for
    causal language modeling training.
    
    Key features:
    - Validates tokenizer compatibility via hash matching
    - Filters sequences by maximum length constraints
    - Adds BOS/EOS tokens automatically
    - Masks metadata tokens in labels (sets to -100)
    - Optional sorting by sequence length for efficient batching
    - Proper label alignment for causal language modeling
    
    The dataset expects JSON files with the structure:
    {
        "input_ids": [token_ids...],
        "labels": [label_ids...], 
        "tokenizer_hash": "...",
        "metadata": {...}
    }
    
    Args:
        files_paths: Sequence of paths to JSON files containing tokenized data
        tokeniser: MyTokeniser instance used to generate the token files
        max_sequence_length: Maximum allowed sequence length (includes BOS/EOS)
        bos_token_id: Token ID for beginning of sequence marker
        eos_token_id: Token ID for end of sequence marker  
        pad_token_id: Token ID for padding (stored but not used in this dataset)
        sort_by_length: Whether to sort files by sequence length for efficient batching
        
    Raises:
        TypeError: If tokeniser is not a MyTokeniser instance
        ValueError: If no compatible files are found after filtering
        
    Note:
        - Only loads files with matching tokenizer hash for compatibility
        - Sequences exceeding max_sequence_length (minus 2 for BOS/EOS) are filtered out
        - Labels are padded with -100 for metadata tokens and BOS/EOS to exclude from loss
        - Sorting by length reduces padding waste during batch training
        
    Example:
        >>> dataset = MyTokenDataset(
        ...     files_paths=Path("tokens").glob("*.json"),
        ...     tokeniser=my_tokenizer,
        ...     max_sequence_length=512,
        ...     bos_token_id=0,
        ...     eos_token_id=1,
        ...     pad_token_id=2
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]  # Returns dict with 'input_ids' and 'labels' tensors
    """
    
    def __init__(self, files_paths: Sequence[Path], tokeniser: MyTokeniser, max_sequence_length: int, bos_token_id: int, eos_token_id: int, pad_token_id: int, sort_by_length: bool = True):
        if not isinstance(tokeniser, MyTokeniser):
            raise TypeError(f"Expected tokeniser to be MyTokeniser (thats what this dataloader is designed for), got {type(tokeniser)}")

        self.tokeniser = tokeniser 
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.files_paths = [
            file for file in files_paths
            if (temp := json.loads(file.read_text()))[constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY] == self.tokeniser.hexa_hash and
               len(temp[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY]) <= max_sequence_length - 2 # Account for BOS and EOS tokens
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



