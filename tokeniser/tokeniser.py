import music21, miditok, constants as constants, math
from typing import *
from functools import cached_property, wraps
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Metadata:
    # weights for complexity measures
    DENSITY_COMPLEXITY_WEIGHT = constants.tokeniser_constants.DENSITY_COMPLEXITY_WEIGHT
    DURATION_COMPLEXITY_WEIGHT = constants.tokeniser_constants.DURATION_COMPLEXITY_WEIGHT
    INTERVAL_COMPLEXITY_WEIGHT = constants.tokeniser_constants.INTERVAL_COMPLEXITY_WEIGHT

    score: music21.stream.base.Score 

    def __post_init__(self):
        if len(self.score.parts) != 2:
            raise ValueError("Expected two staves (RH and LH), got something else.")
        self.key_signatures
        self.time_signatures
        self.num_measures

    @cached_property
    def rh_part(self) -> music21.stream.base.Part:
        return self.score.parts[0]
    
    @cached_property
    def lh_part(self) -> music21.stream.base.Part:
        return self.score.parts[1]
    
    @cached_property
    def tempos(self) -> List[float]:
        tempos = self.rh_part.recurse().getElementsByClass(music21.tempo.MetronomeMark)
        return [t for tempo in tempos if (t := tempo.getQuarterBPM())] if tempos else [80] 

    @cached_property
    def key_signatures(self) -> List[music21.key.KeySignature]:
        signatures = list(self.rh_part.recurse().getElementsByClass(music21.key.KeySignature))
         
        if not signatures:
            raise ValueError("No key signatures found in the score.")
        return signatures

    @cached_property
    def time_signatures(self) -> List[str]:
        signatures = self.rh_part.recurse().getElementsByClass(music21.meter.TimeSignature)
        res = [signature.ratioString for signature in signatures]
        if not res:
            raise ValueError("No time signatures found in the score.")
        return res 
        

    @cached_property
    def num_measures(self) -> int:
        try:
            score = music21.repeat.Expander(self.rh_part).process()
        except Exception as e:
            score = self.score
        
        return len(score.getElementsByClass(music21.stream.Measure))
    
    @cached_property
    def rh_clefs(self) -> List[str]:
        return [clef.sign for clef in self.rh_part.recurse().getElementsByClass(music21.clef.Clef) if clef.sign]
    
    @cached_property
    def lh_clefs(self) -> List[str]:
        return [clef.sign for clef in self.lh_part.recurse().getElementsByClass(music21.clef.Clef) if clef.sign]
    
    @cached_property
    def rh_notes(self) -> List[music21.note.NotRest]:
        return [note for note in self.rh_part.flatten().notes]

    @cached_property
    def lh_notes(self) -> List[music21.note.NotRest]:
        return [note for note in self.lh_part.flatten().notes]
    
    @cached_property
    def rh_midi_values(self) -> List[int]:
        midi_values: List[int] = []
        for n in self.rh_notes:
            if isinstance(n, music21.note.Note):
                midi_values.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                midi_values.extend(p.midi for p in n.pitches)  # all pitches in the chord
        return midi_values
    
    @cached_property
    def lh_midi_values(self) -> List[int]:
        midi_values: List[int] = []
        for n in self.lh_notes:
            if isinstance(n, music21.note.Note):
                midi_values.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                midi_values.extend(p.midi for p in n.pitches)  # all pitches in the chord
        return midi_values
    
    @cached_property
    def notes(self) -> List[music21.note.NotRest]:
        return self.rh_notes + self.lh_notes  # combine notes from both parts
    
    @cached_property
    def midi_values(self) -> List[int]:
        return self.rh_midi_values + self.lh_midi_values  # combine MIDI values from both parts
    
    @cached_property
    def num_total_notes(self) -> int:
        return len(self.notes)

    @cached_property
    def pitch_range(self) -> Tuple[music21.pitch.Pitch, music21.pitch.Pitch]:
        if self.midi_values:
            pitch_min = music21.pitch.Pitch(midi=min(self.midi_values))
            pitch_max = music21.pitch.Pitch(midi=max(self.midi_values))
            return (pitch_min, pitch_max)

        return (music21.pitch.Pitch("C4"), music21.pitch.Pitch("C5"))  # fallback

    
    @cached_property
    def rh_intervals(self) -> List[int]:
        return [abs(n2 - n1) for n1, n2 in zip(self.rh_midi_values[:-1], self.rh_midi_values[1:])]
    
    @cached_property
    def lh_intervals(self) -> List[int]:
        return [abs(n2 - n1) for n1, n2 in zip(self.lh_midi_values[:-1], self.lh_midi_values[1:])]  # intervals for left hand only
    
    @cached_property
    def intervals(self) -> List[int]:
        return self.rh_intervals + self.lh_intervals  # combine intervals from both hands
    
    @cached_property
    def density_complexity(self) -> int:
        note_density = Metadata.DENSITY_COMPLEXITY_WEIGHT * (self.num_total_notes / max(self.num_measures, 1) / 2)
        return min(10, math.ceil(max(1, note_density))) # Complexity (placeholder heuristic: density)

    @cached_property
    def duration_complexity(self) -> int:
        unique_durations =  set(round(n.quarterLength, 2) for n in self.notes)
        return min(10, math.ceil(max(1, Metadata.DURATION_COMPLEXITY_WEIGHT * len(unique_durations))))  # Duration complexity (how many types of durations are used?)

    @cached_property
    def interval_complexity(self) -> int:
        avg_interval = Metadata.INTERVAL_COMPLEXITY_WEIGHT * sum(self.intervals) / len(self.intervals) if self.intervals else 1
        return min(10, math.ceil(avg_interval / 2))  # Pitch complexity (based on interval size variability)

    # deprecated, use data property instead
    @cached_property
    def data(self) -> Dict[str, Any]:
        return {
            "key_signature": self.key_signatures[0].sharps,
            "time_signature": self.time_signatures[0],
            "clefs": {"RH": self.rh_clefs[0], "LH": self.lh_clefs[0]},
            "pitch_range": self.pitch_range,
            "num_measures": self.num_measures,
            "density_complexity": self.density_complexity,
            "duration_complexity": self.duration_complexity,
            "pitch_complexity": self.interval_complexity
        }

    @cached_property
    def tokenised_metadata(self) -> "TokenisedMetadata":
        """
        Returns a TokenisedMetadata object containing the metadata as tokenised strings.
        This is useful for encoding the metadata into a format suitable for the tokeniser.
        """
        return self.TokenisedMetadata(
            key_signature=self.key_signatures[0].sharps,
            time_signature=self.time_signatures[0],
            rh_clef=self.rh_clefs[0],
            lh_clef=self.lh_clefs[0],
            lowest_pitch=self.pitch_range[0].midi,
            highest_pitch=self.pitch_range[1].midi,
            num_measures=self.num_measures,
            density_complexity=self.density_complexity,
            duration_complexity=self.duration_complexity,
            interval_complexity=self.interval_complexity
        )

    @dataclass
    class TokenisedMetadata:
        key_signature: int
        time_signature: str
        rh_clef: str
        lh_clef: str
        lowest_pitch: int
        highest_pitch: int
        num_measures: int
        density_complexity: int
        duration_complexity: int
        interval_complexity: int

        def to_dict(self) -> Dict[str, int]:
            return {
                constants.tokeniser_constants.KEY_SIGNATURE_FIELD: f"{constants.tokeniser_constants.KEY_SIG_TOKEN_PREFIX}{self.key_signature}",
                constants.tokeniser_constants.TIME_SIGNATURE_FIELD: f"TimeSig_{self.time_signature}",
                constants.tokeniser_constants.RH_CLEF_FIELD: f"{constants.tokeniser_constants.CLEF_TOKEN_PREFIX}{self.rh_clef}",
                constants.tokeniser_constants.LH_CLEF_FIELD: f"{constants.tokeniser_constants.CLEF_TOKEN_PREFIX}{self.lh_clef}",
                constants.tokeniser_constants.LOWEST_PITCH_FIELD: f"Pitch_{self.lowest_pitch}",
                constants.tokeniser_constants.HIGHEST_PITCH_FIELD: f"Pitch_{self.highest_pitch}",
                constants.tokeniser_constants.NUM_MEASURES_FIELD: f"{constants.tokeniser_constants.BAR_TOKEN_PREFIX}{self.num_measures}",
                constants.tokeniser_constants.DENSITY_COMPLEXITY_FIELD: f"{constants.tokeniser_constants.DENSITY_COMPL_TOKEN_PREFIX}{self.density_complexity}",
                constants.tokeniser_constants.DURATION_COMPLEXITY_FIELD: f"{constants.tokeniser_constants.DURATION_COMPL_TOKEN_PREFIX}{self.duration_complexity}",
                constants.tokeniser_constants.INTERVAL_COMPLEXITY_FIELD: f"{constants.tokeniser_constants.INTERVAL_COMPL_TOKEN_PREFIX}{self.interval_complexity}"
            }
        
        def to_list(self) -> List[str]:
            return list(self.to_dict().values())
        
        @staticmethod
        def type_to_dict(input: "Metadata | Metadata.TokenisedMetadata | Dict[str, str]") -> Dict[str, int]:
            """
            Converts Metadata or TokenisedMetadata to a dictionary.
            If input is already a dictionary, it returns it as is.
            """
            if isinstance(input, Metadata):
                return input.tokenised_metadata.to_dict()
            elif isinstance(input, Metadata.TokenisedMetadata):
                return input.to_dict()
            elif isinstance(input, dict):
                return input
            else:
                raise TypeError(f"Expected Metadata or TokenisedMetadata, got {type(input)}")

class MyTokeniserConfig(miditok.classes.TokenizerConfig):
        """
        Custom configuration for the MyTokeniser.
        This class inherits from miditok.classes.TokenizerConfig and can be extended if needed.
        """
    
        def __init__(self, 
                     time_signature_range: dict[int, list[int]] = None,
                     max_bars: int = None):
            """
            Initialize the MyTokeniserConfig with custom parameters.
            :param time_signature_range: Dictionary defining valid time signatures.
            :param pitch_range: Tuple defining the MIDI pitch range.
            :param max_bars: Maximum number of bars for embedding.
            """
            
            config = constants.tokeniser_constants.MYTOKENISER_BASE_CONFIG.copy()
            if time_signature_range:
                config[constants.tokeniser_constants.TIME_SIGNATURE_RANGE_FIELD] = time_signature_range

            if max_bars:
                config[constants.tokeniser_constants.MAX_BARS_FIELD] = max_bars

            super().__init__(**config)
                   
class MyTokeniser(miditok.REMI):
    """
    Custom tokeniser class that extends miditok.REMI.
    This class is used to build a tokeniser for MIDI files using the REMI scheme.
    """ 
    
    def __init__(self, tokenizer_config: MyTokeniserConfig = MyTokeniserConfig(), params: Optional[Path] = None):
        if not isinstance(tokenizer_config, MyTokeniserConfig):
            raise TypeError(f"Expected MyTokeniserConfig, got {type(tokenizer_config)}")

        super().__init__(tokenizer_config=tokenizer_config, params=params)

        self.bos_token = constants.tokeniser_constants.BOS_TOKEN
        self.eos_token = constants.tokeniser_constants.EOS_TOKEN
        self.pad_token = constants.tokeniser_constants.PAD_TOKEN

        # trained tokenisers should have their whole vocab initialised again,
        # untrained ones get theirs created again, without our metadata tokens        

        if not self.is_trained:
            self.add_key_signatures_to_vocab()
            self.add_clefs_to_vocab()
            self.add_complexities_to_vocab()
            self.add_bars_to_vocab()


    def encode_with_metadata(self, input_file: Path, tokenised_metadata: Metadata | Metadata.TokenisedMetadata | dict) -> dict:
        metadata_seq = self.encode_metadata(tokenised_metadata)

        # encode_ids=true only encodes ids only if tokeniser is trained
        tok_seq = self.encode(input_file, encode_ids=True, no_preprocess_score=False, attribute_controls_indexes=None)


        return {constants.tokeniser_constants.TOKENS_METADATA_KEY: metadata_seq.tokens,
                constants.tokeniser_constants.TOKENS_TOKENS_KEY: tok_seq.tokens,
                constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY: metadata_seq.ids + tok_seq.ids,
                constants.tokeniser_constants.TOKENS_LABELS_KEY: tok_seq.ids,
                constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY: self.hexa_hash}

    # TODO: rename
    def encode_metadata(self, tokenised_metadata: Metadata | Metadata.TokenisedMetadata | dict) -> miditok.TokSequence:
        tokenised_metadata = Metadata.TokenisedMetadata.type_to_dict(tokenised_metadata)
        
        metadata_seq = miditok.TokSequence(list(tokenised_metadata.values()))
        self.complete_sequence(metadata_seq, complete_bytes=True)
        if self.is_trained:
            self.encode_token_ids(metadata_seq)

        return metadata_seq

    def save_generated_tokens(self, output_file: Path, generated_tokens: list[int], metadata: Metadata | Metadata.TokenisedMetadata | dict) -> None:
        import json

        if isinstance(metadata, Metadata):
            metadata = metadata.tokenised_metadata.to_dict()
        elif isinstance(metadata, Metadata.TokenisedMetadata):
            metadata = metadata.to_dict()

        with open(output_file, "w") as f:
            json.dump({
                constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY: generated_tokens,
                constants.tokeniser_constants.TOKENS_METADATA_KEY: metadata,
                constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY: self.hexa_hash
            }, f, indent=4)

    def train_BPE(self, data_dir: Path):
        """
        Train the tokeniser on a dir of MIDI files.
        This method will build the vocabulary based on the MIDI files in the specified dir.
        """
        # from data_pipeline_scripts.conversion_functions import Generics

        files = list(data_dir.glob(f"*{constants.MIDI_EXTENSION}"))

        print(f"Training BPE on {len(files)} {constants.MIDI_EXTENSION} files in {data_dir}")

        self.train(vocab_size=self.vocab_size * constants.tokeniser_constants.BPE_VOCAB_SCALE_FACTOR, model='BPE', iterator=miditok.tokenizer_training_iterator.TokTrainingIterator(self, files))

        # Generics.clear_n_terminal_lines(2)
    

    def add_key_signatures_to_vocab(self) -> None:
        for i in range(-7, 8):
            self.add_to_vocab(constants.tokeniser_constants.KEY_SIG_TOKEN_PREFIX + str(i))


    def add_clefs_to_vocab(self) -> None:
        for clef in self.config.additional_params[constants.tokeniser_constants.CLEF_FIELD_NAME]:
            self.add_to_vocab(constants.tokeniser_constants.CLEF_TOKEN_PREFIX + clef)


    def add_complexities_to_vocab(self) -> None:
        for i in range(1, 11):
            self.add_to_vocab(constants.tokeniser_constants.DENSITY_COMPL_TOKEN_PREFIX + str(i))
            self.add_to_vocab(constants.tokeniser_constants.DURATION_COMPL_TOKEN_PREFIX + str(i))
            self.add_to_vocab(constants.tokeniser_constants.INTERVAL_COMPL_TOKEN_PREFIX + str(i))


    def add_bars_to_vocab(self) -> None:
        for i in range(1, self.config.additional_params[constants.tokeniser_constants.MAX_BARS_FIELD] + 1):
            self.add_to_vocab(constants.tokeniser_constants.BAR_TOKEN_PREFIX + str(i))

    # TODO: Maybe find a way to prevent false negatives, but wed need to extract which lists in the config have a predifined order
    # and which ones are just sets like the clefs, time signatures,
    @property
    def hexa_hash(self) -> str:
        """
        Hash is based on unsorted tokeniser json as returned by to_dict() !!! allows for false negatives !!!
        """
        
        import hashlib

        return hashlib.sha256(str(self.to_dict()).encode('utf-8')).hexdigest()


    def valid_metadata(self, tokenised_data: Metadata | Metadata.TokenisedMetadata | Dict[str, str]) -> Tuple[bool, str]:
        res = ""

        tokenised_data = Metadata.TokenisedMetadata.type_to_dict(tokenised_data)

        if (t := int(tokenised_data[constants.tokeniser_constants.NUM_MEASURES_FIELD].replace(constants.tokeniser_constants.BAR_TOKEN_PREFIX, ""))) > self.config.additional_params[constants.tokeniser_constants.MAX_BARS_FIELD]:
            res += f"Invalid number of measures: {t} not in range 1-{self.config.additional_params[constants.tokeniser_constants.MAX_BARS_FIELD]} Bars\n"

        if (t := tokenised_data[constants.tokeniser_constants.TIME_SIGNATURE_FIELD]) not in self.vocab:
            res += f"Invalid time signature: {t} not in {self.config.time_signature_range} Time Signatures\n"

        if (c := tokenised_data[constants.tokeniser_constants.RH_CLEF_FIELD]) not in self.vocab:
            res += f"Invalid RH clef: {c} not in {self.config.additional_params[constants.tokeniser_constants.CLEF_FIELD_NAME]} Clefs\n"

        if (c := tokenised_data[constants.tokeniser_constants.LH_CLEF_FIELD]) not in self.vocab:
            res += f"Invalid LH clef: {c} not in {self.config.additional_params[constants.tokeniser_constants.CLEF_FIELD_NAME]} Clefs\n"

        if (l := tokenised_data[constants.tokeniser_constants.LOWEST_PITCH_FIELD]) not in self.vocab:
            res += f"Invalid lowest pitch: {l} not in MIDI range {self.config.pitch_range}\n"

        if (h := tokenised_data[constants.tokeniser_constants.HIGHEST_PITCH_FIELD]) not in self.vocab:
            res += f"Invalid highest pitch: {h} not in MIDI range {self.config.pitch_range}\n"

        if res:
            return False, res
        
        return True, ""

    @classmethod
    def from_pretrained(
        cls: Type["MyTokeniser"],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **model_kwargs,
    ) -> "MyTokeniser":
        

        tokeniser = super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **model_kwargs,
        )
        if not isinstance(tokeniser, MyTokeniser):
            raise ValueError(f"The loaded tokeniser is not an instance of MyTokeniser, but {type(tokeniser)}")
        return tokeniser

    def _load_from_json(self, file_path):
        import json
        # this function is called when an instance is created from a JSON file via the params argument of the init,
        # here we just check that the loaded tokeniser is an instance of MyTokeniser
        
        with Path(file_path).open() as param_file:
            params = json.load(param_file)

        if params["tokenization"] != self.__class__.__name__:
            raise ValueError(f"The loaded tokeniser is not an instance of MyTokeniser, but {params['tokenization']}")
        
        super()._load_from_json(file_path)
    

if __name__ == "__main__":
    pass
    #print(tokeniser.vocab_model)
    # print(tokeniser.config.to_dict())

