import miditok.constants # type: ignore
from typing import *

# token format
TOKENS_INPUT_IDS_KEY: str = "input_ids"
TOKENS_LABELS_KEY: str = "labels"
TOKENS_TOKENISER_HASH_KEY: str = "tokeniser_hash"
TOKENS_METADATA_KEY: str = "metadata"
TOKENS_TOKENS_KEY: str = "tokens"
TOKENS_KEY_SIGNATURE_KEY: str = "desired_key_signature"


# Constants for tokeniser configuration
TIME_SIGNATURE_RANGE_FIELD: str = "time_signature_range"
MAX_BARS_FIELD: str = "max_bars" # dont set this to 'max_bar_embedding'(that will make the tokeniser )
METADATA_LENGTH_FIELD: str = "metadata_length"


MODIFYABLE_TOKENISER_CONFIG_FIELDS: dict[str, Any] = {
    TIME_SIGNATURE_RANGE_FIELD: {8: [3, 12, 6, 9], 4: [5, 6, 3, 2, 1, 4], 2: [1, 2, 3, 4]},
    MAX_BARS_FIELD: 52
}


MYTOKENISER_BASE_CONFIG: dict[str, Any] = {
            "use_programs": True,
            "use_time_signatures": True,
            # "use_chords": True,
            "use_rests": True,
            "use_pitch_intervals": True,
            # "chord_tokens_with_root_note": True,
            # "chord_unknown": (2, 4),
            "one_token_stream_for_programs": True,
            "clefs": ['G', 'F'],
            **MODIFYABLE_TOKENISER_CONFIG_FIELDS
        }


CLEF_FIELD_NAME: str = "clefs"


PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN  = [f"{token}_None" for token in miditok.constants.SPECIAL_TOKENS]

# Token prefixes
BAR_TOKEN_PREFIX: str = "Bar_"
KEY_SIG_TOKEN_PREFIX: str = "KeySig_"
CLEF_TOKEN_PREFIX: str = "Clef_"
DENSITY_COMPL_TOKEN_PREFIX: str = "Dens_"
DURATION_COMPL_TOKEN_PREFIX: str = "Dur_"
INTERVAL_COMPL_TOKEN_PREFIX: str = "Int_"


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




