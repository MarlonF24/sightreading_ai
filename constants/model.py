from typing import *
from .tokeniser import TOKENS_TOKENISER_HASH_KEY

# Model constants
TRAINING_DIR_NAME: str = "training"
LOGS_DIR_NAME: str = "logs"
OUTPUT_DIR_NAME: str = "output"


TOKENISER_HASH_FIELD: str = TOKENS_TOKENISER_HASH_KEY
VOCAB_SIZE_FIELD: str = "vocab_size"
BOS_TOKEN_ID_FIELD: str = "bos_token_id"
EOS_TOKEN_ID_FIELD: str = "eos_token_id"
PAD_TOKEN_ID_FIELD: str = "pad_token_id"

MODIFYABLE_MODEL_CONFIG_FIELDS: dict[str, Any] = {
    TOKENISER_HASH_FIELD: 0,
    VOCAB_SIZE_FIELD: 0,
    BOS_TOKEN_ID_FIELD: 0,
    EOS_TOKEN_ID_FIELD: 0,
    PAD_TOKEN_ID_FIELD: 0,
}

MYMODEL_BASE_CONFIG: dict[str, Any] = {
    "architectures": ["MyModel"],
    "n_embd": 512,
    "n_layer": 6,
    "n_head": 8,
    **MODIFYABLE_MODEL_CONFIG_FIELDS
}



