import miditok.constants
from typing import *

# converter constants
CONVERTER_PIPELINE_DIR_NAME: str = "data_pipeline"
CONVERTER_DATA_DIR_DEFAULT_NAME: str = "data"
CONVERTER_LOGS_DIR_NAME: str = "logs"

# conversion functions constants
METADATA_DIR_NAME: str = "metadata_files"

# token format
INPUT_IDS_KEY: str = "input_ids"
LABELS_KEY: str = "labels"
TOKENISER_HASH_KEY: str = "tokeniser_hash"
METADATA_KEY: str = "metadata"

TIME_SIGNATURE_RANGE_FIELD: str = "time_signature_range"
MAX_BARS_FIELD: str = "max_bar_embedding"

# Constants for tokeniser configuration
MODIFYABLE_TOKENISER_CONFIG_FIELDS: dict[str, Any] = {
    TIME_SIGNATURE_RANGE_FIELD: {8: [3, 12, 6, 9], 4: [5, 6, 3, 2, 1, 4], 2: [1, 2, 3, 4]},
    MAX_BARS_FIELD: 33
}

MYTOKENISER_BASE_CONFIG: dict[str, Any] = {
            "use_programs": True,
            "use_time_signatures": True,
            # "use_chords": True,
            "use_rests": True,
            # "chord_tokens_with_root_note": True,
            # "chord_unknown": (2, 4),
            "one_token_stream_for_programs": True,
            "clefs": ['G', 'F'],
            **MODIFYABLE_TOKENISER_CONFIG_FIELDS
        }


CLEF_FIELD_NAME: str = "clefs"


PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN  = [f"{token}_None" for token in miditok.constants.SPECIAL_TOKENS]


# Token prefixes
KEY_SIG_TOKEN_PREFIX: str = "KeySig_"
CLEF_TOKEN_PREFIX: str = "Clef_"
DENSITY_COMPL_TOKEN_PREFIX: str = "Dens_"
DURATION_COMPL_TOKEN_PREFIX: str = "Dur_"
INTERVAL_COMPL_TOKEN_PREFIX: str = "Int_"

# File extensions
MIDI_EXTENSION: str = ".midi"
PDF_EXTENSION: str = ".pdf"
MXL_EXTENSION: str = ".mxl"
MUSICXML_EXTENSION: str = ".musicxml"
JSON_EXTENSION: str = ".json"
LOG_EXTENSION: str = ".log"
JAR_EXTENSION: str = ".jar"
TOKENS_EXTENSION: str = ".tokens.json"
METADATA_EXTENSION: str = ".metadata.json"

# BPE 
BPE_VOCAB_SCALE_FACTOR: int = 2

# Metadata
DENSITY_COMPLEXITY_WEIGHT: int = 1
DURATION_COMPLEXITY_WEIGHT: int = 1
INTERVAL_COMPLEXITY_WEIGHT: int = 1

KEY_SIGNATURE_FIELD: str = "key_signature"
TIME_SIGNATURE_FIELD: str = "time_signature"
RH_CLEF_FIELD: str = "rh_clef"
LH_CLEF_FIELD: str = "lh_clef"
LOWEST_PITCH_FIELD: str = "lowest_pitch"
HIGHEST_PITCH_FIELD: str = "highest_pitch"
NUM_MEASURES_FIELD: str = "num_measures"
DENSITY_COMPLEXITY_FIELD: str = "density_complexity"
DURATION_COMPLEXITY_FIELD: str = "duration_complexity"
INTERVAL_COMPLEXITY_FIELD: str = "interval_complexity"

# Model constants
TRAINING_DIR_NAME: str = "training"
LOGS_DIR_NAME: str = "logs"
OUTPUT_DIR_NAME: str = "output"

MYMODEL_BASE_CONFIG: dict[str, Any] = {
    "architectures": ["MyModel"],
    "n_embd": 512,
    "n_layer": 6,
    "n_head": 8,
    **MODIFYABLE_TOKENISER_CONFIG_FIELDS
}

TOKENISER_HASH_FIELD: str = TOKENISER_HASH_KEY
VOCAB_SIZE_FIELD: str = "vocab_size"
BOS_TOKEN_ID_FIELD: str = "bos_token_id"
EOS_TOKEN_ID_FIELD: str = "eos_token_id"
PAD_TOKEN_ID_FIELD: str = "pad_token_id"

MODIFYABLE_TOKENISER_CONFIG_FIELDS: dict[str, Any] = {
    TOKENISER_HASH_FIELD: 0,
    VOCAB_SIZE_FIELD: 0,
    BOS_TOKEN_ID_FIELD: 0,
    EOS_TOKEN_ID_FIELD: 0,
    PAD_TOKEN_ID_FIELD: 0,
}
