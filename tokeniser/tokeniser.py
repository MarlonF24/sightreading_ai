import music21, miditok, constants as constants, math
import music21.meter.base
from typing import *
from functools import cached_property, wraps
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Metadata:
    """
    Extracts and computes metadata from music21 Score objects for tokenization.
    
    This class analyzes musical scores to extract structural and complexity metrics
    that are used as conditioning tokens in machine learning models. It handles
    piano scores with separate right-hand and left-hand parts, computing various
    complexity measures and musical characteristics.
    
    The class automatically validates input scores and fills in missing musical
    elements (key signatures, time signatures) during initialization.
    
    Attributes:
        DENSITY_COMPLEXITY_WEIGHT: Weight factor for note density calculations
        DURATION_COMPLEXITY_WEIGHT: Weight factor for duration variety calculations  
        INTERVAL_COMPLEXITY_WEIGHT: Weight factor for interval complexity calculations
        score: The music21 Score object to analyze
        
    Raises:
        ValueError: If score doesn't have exactly 2 piano parts (RH and LH)
        
    Example:
        >>> score = music21.converter.parse("piece.musicxml")
        >>> metadata = Metadata(score)
        >>> complexity = metadata.density_complexity
    """
    
    # weights for complexity measures
    DENSITY_COMPLEXITY_WEIGHT = constants.tokeniser_constants.DENSITY_COMPLEXITY_WEIGHT
    DURATION_COMPLEXITY_WEIGHT = constants.tokeniser_constants.DURATION_COMPLEXITY_WEIGHT
    INTERVAL_COMPLEXITY_WEIGHT = constants.tokeniser_constants.INTERVAL_COMPLEXITY_WEIGHT

    score: music21.stream.Score 

    def __post_init__(self):
        """
        Validates score structure and ensures required musical elements exist.
        
        Checks that the score has exactly 2 piano parts and triggers computation
        of time signatures and key signatures, filling in missing elements if needed.
        """
        instrument_lists = [list(part.getInstruments()) for part in self.score.parts]

        if (temp := len(self.score.parts)) != 2 or any(not isinstance(instrument, (music21.instrument.Piano, music21.instrument.ElectricPiano)) for instrument_list in instrument_lists for instrument in instrument_list):
            raise ValueError(f"Metadata expects scores with two piano staves (RH and LH), got {temp} staves with instruments {instrument_lists}.")

        self.time_signatures
        self.key_signatures

    @cached_property
    def rh_part(self) -> music21.stream.Part:
        return self.score.parts[0]
    
    @cached_property
    def lh_part(self) -> music21.stream.Part:
        return self.score.parts[1]
    
    @cached_property
    def tempos(self) -> List[float]:
        tempos = self.score.recurse().getElementsByClass(music21.tempo.MetronomeMark)
        return [t for tempo in tempos if (t := tempo.getQuarterBPM())] if tempos else [80] 

    @cached_property
    def rh_measures(self) -> Sequence[music21.stream.Measure]:
        return self.rh_part.getElementsByClass(music21.stream.Measure)

    @cached_property
    def lh_measures(self) -> Sequence[music21.stream.Measure]:
        return self.lh_part.getElementsByClass(music21.stream.Measure)

    @cached_property
    def key_signatures(self) -> Sequence[music21.key.KeySignature]:
        # Note: this can be corruptive if the score has multple key signatures but not the first one was returned
        signatures = list(self.score.recurse().getElementsByClass(music21.key.KeySignature))
         
        if not signatures:
            signatures = [self.score.analyze('key')]

            if not self.score.keySignature:
                self.score.keySignature = signatures[0]

            if not self.rh_measures[0].keySignature:
                self.rh_measures[0].keySignature = signatures[0]

            if not self.lh_measures[0].keySignature:
                self.lh_measures[0].keySignature = signatures[0]

        return signatures

    @cached_property
    def time_signatures(self) -> List[music21.meter.base.TimeSignature]:
        # Note: this can be corruptive if the score has multple time signatures but not the first one was returned
        signatures = list(self.score.recurse().getElementsByClass(music21.meter.base.TimeSignature))

        # If no time signatures are found, fall back to the best time signature of each measure this can also be corruptive, as it could ignore any time signature changes
        
        if not signatures:
            signature_list = []


            for measure in list(self.rh_measures) + list(self.lh_measures):
                try:
                    signature_list.append(measure.bestTimeSignature())
                except music21.exceptions21.MeterException as e:
                    pass
                
            signatures = [max(set(signature_list), key=signature_list.count)]

            if not self.score.timeSignature:
                self.score.timeSignature = signatures[0]

            if not self.rh_measures[0].timeSignature:
                self.rh_measures[0].timeSignature = signatures[0]
            if not self.lh_measures[0].timeSignature:
                self.lh_measures[0].timeSignature = signatures[0]
        

        return signatures

    @cached_property
    def num_measures(self) -> int:
        return len(self.rh_part.getElementsByClass(music21.stream.Measure))

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
        return [abs(n2 - n1) for n1, n2 in zip(self.lh_midi_values[:-1], self.lh_midi_values[1:])]  
    
    @cached_property
    def intervals(self) -> List[int]:
        return self.rh_intervals + self.lh_intervals  # combine intervals from both hands
    
    @cached_property
    def density_complexity(self) -> int:
        """
        Calculate note density complexity on a scale of 1-10.
        
        Measures how densely packed notes are relative to the number of measures,
        normalized for two-handed piano playing. Higher values indicate more
        notes per measure.
        
        Returns:
            Complexity score from 1-10, capped at 10
        """
        note_density = Metadata.DENSITY_COMPLEXITY_WEIGHT * (self.num_total_notes / max(self.num_measures, 1) / 2)
        return min(10, math.ceil(max(1, note_density)))

    @cached_property
    def duration_complexity(self) -> int:
        """
        Calculate rhythmic complexity based on variety of note durations.
        
        Measures how many different note durations are used in the piece.
        More variety in note lengths indicates higher rhythmic complexity.
        
        Returns:
            Complexity score from 1-10, capped at 10
        """
        unique_durations =  set(round(n.quarterLength, 2) for n in self.notes)
        return min(10, math.ceil(max(1, Metadata.DURATION_COMPLEXITY_WEIGHT * len(unique_durations))))

    @cached_property
    def interval_complexity(self) -> int:
        """
        Calculate melodic complexity based on average interval sizes.
        
        Measures the average size of melodic intervals (pitch jumps) between
        consecutive notes. Larger intervals indicate more complex melodies.
        
        Returns:
            Complexity score from 1-10, capped at 10
        """
        avg_interval = Metadata.INTERVAL_COMPLEXITY_WEIGHT * sum(self.intervals) / len(self.intervals) if self.intervals else 1
        return min(10, math.ceil(avg_interval / 2))

    @cached_property
    def tokenised_metadata(self) -> "TokenisedMetadata":
        """
        Convert metadata to tokenized format suitable for machine learning models.
        
        Returns:
            TokenisedMetadata object with string-formatted metadata tokens
        """
        return self.TokenisedMetadata(
            time_signature=self.time_signatures[0].ratioString,
            num_measures=self.num_measures,
            density_complexity=self.density_complexity,
            duration_complexity=self.duration_complexity,
            interval_complexity=self.interval_complexity
        )

    @dataclass
    class TokenisedMetadata:
        """
        Container for metadata formatted as tokenizer-compatible strings.
        
        Converts numerical and musical metadata into string tokens that can be
        used as conditioning inputs for music generation models.
        
        Attributes:
            time_signature: Time signature as ratio string (e.g., "4/4")
            num_measures: Number of measures in the piece
            density_complexity: Note density complexity (1-10)
            duration_complexity: Rhythmic complexity (1-10)
            interval_complexity: Melodic complexity (1-10)
        """
        time_signature: str
        num_measures: int
        density_complexity: int
        duration_complexity: int
        interval_complexity: int

        @property
        def as_dict(self) -> Dict[str, str]:
            """
            Convert tokenized metadata to dictionary with prefixed token strings.
            
            Returns:
                Dictionary mapping field names to properly formatted token strings
            """
            return {
                constants.tokeniser_constants.TIME_SIGNATURE_FIELD: f"TimeSig_{self.time_signature}",
                constants.tokeniser_constants.NUM_MEASURES_FIELD: f"{constants.tokeniser_constants.BAR_TOKEN_PREFIX}{self.num_measures}",
                constants.tokeniser_constants.DENSITY_COMPLEXITY_FIELD: f"{constants.tokeniser_constants.DENSITY_COMPL_TOKEN_PREFIX}{self.density_complexity}",
                constants.tokeniser_constants.DURATION_COMPLEXITY_FIELD: f"{constants.tokeniser_constants.DURATION_COMPL_TOKEN_PREFIX}{self.duration_complexity}",
                constants.tokeniser_constants.INTERVAL_COMPLEXITY_FIELD: f"{constants.tokeniser_constants.INTERVAL_COMPL_TOKEN_PREFIX}{self.interval_complexity}"
            }

        @property
        def as_list(self) -> List[str]:
            """
            Convert tokenized metadata to list of token strings.
            
            Returns:
                List of formatted token strings in consistent order
            """
            return list(self.as_dict.values())

        @staticmethod
        def type_to_dict(input: "Metadata | Metadata.TokenisedMetadata | Dict[str, str]") -> Dict[str, str]:
            """
            Normalize different metadata input types to dictionary format.
            
            Args:
                input: Metadata object, TokenisedMetadata object, or dictionary
                
            Returns:
                Dictionary with tokenized metadata strings
                
            Raises:
                TypeError: If input type is not supported
            """
            if isinstance(input, Metadata):
                return input.tokenised_metadata.as_dict
            elif isinstance(input, Metadata.TokenisedMetadata):
                return input.as_dict
            elif isinstance(input, dict):
                return input
            else:
                raise TypeError(f"Expected Metadata or TokenisedMetadata, got {type(input)}")

class MyTokeniserConfig(miditok.classes.TokenizerConfig):
    """
    Custom configuration class for MyTokeniser extending miditok's base configuration.
    
    Provides specialized configuration for piano-focused tokenization with metadata
    conditioning. Inherits from miditok.classes.TokenizerConfig and adds custom
    parameters for time signatures and measure limits.
    
    The configuration is built from base constants and can be customized for
    specific model requirements or dataset characteristics.
    """
    
    def __init__(self, 
                 time_signature_range: dict[int, list[int]] = None,
                 max_bars: int = None):
        """
        Initialize configuration with custom parameters.
        
        Args:
            time_signature_range: Dictionary mapping numerators to valid denominators
                                for time signatures (e.g., {4: [4], 3: [4, 8]})
            max_bars: Maximum number of bars for measure embeddings
        """
        config = constants.tokeniser_constants.MYTOKENISER_BASE_CONFIG.copy()
        if time_signature_range:
            config[constants.tokeniser_constants.TIME_SIGNATURE_RANGE_FIELD] = time_signature_range

        if max_bars:
            config[constants.tokeniser_constants.MAX_BARS_FIELD] = max_bars
            
        super().__init__(**config)

class MyTokeniser(miditok.REMI):
    """
    Custom MIDI tokenizer extending miditok.REMI with metadata conditioning.
    
    This tokenizer is specifically designed for piano music generation with
    structural and complexity conditioning. It extends the REMI tokenization
    scheme with custom tokens for musical metadata and removes unnecessary
    vocabulary elements for piano-focused applications.
    
    Key features:
    - Metadata conditioning tokens (complexity measures, structure info)
    - Piano-optimized vocabulary (removes drums, unused instruments)
    - BPE training support for vocabulary compression
    - Hash-based tokenizer compatibility checking
    - Validation of metadata against tokenizer constraints
    
    Attributes:
        bos_token: Beginning of sequence token
        eos_token: End of sequence token  
        pad_token: Padding token
        
    Note:
        Trained tokenizers load complete vocabularies, while untrained ones
        build vocabularies dynamically and remove unused tokens.
        
    Example:
        >>> config = MyTokeniserConfig(max_bars=32)
        >>> tokenizer = MyTokeniser(config)
        >>> tokens = tokenizer.encode_with_metadata(midi_file, metadata)
    """
    
    def __init__(self, tokenizer_config: MyTokeniserConfig = MyTokeniserConfig(), params: Optional[Path] = None):
        """
        Initialize the custom tokenizer.
        
        Args:
            tokenizer_config: Configuration object for tokenizer parameters
            params: Path to existing tokenizer parameters file for loading
            
        Raises:
            TypeError: If tokenizer_config is not MyTokeniserConfig instance
        """
        if not isinstance(tokenizer_config, MyTokeniserConfig):
            raise TypeError(f"Expected MyTokeniserConfig, got {type(tokenizer_config)}")

        super().__init__(tokenizer_config=tokenizer_config, params=params)

        self.bos_token = constants.tokeniser_constants.BOS_TOKEN
        self.eos_token = constants.tokeniser_constants.EOS_TOKEN
        self.pad_token = constants.tokeniser_constants.PAD_TOKEN

        # trained tokenisers should have their whole vocab initialised again,
        # untrained ones get theirs created again, without our metadata tokens        

        if not self.is_trained:
            self.add_complexities_to_vocab()
            self.add_bars_to_vocab() # note that there are also bar tokens for tokenisation in miditok, but here we add them for bar length for the metadata
            self.remove_unecessary_program_tokens_from_vocab()
            self.stuff_vocab_index_holes()

    def add_complexities_to_vocab(self) -> None:
        for i in range(1, 11):
            self.add_to_vocab(constants.tokeniser_constants.DENSITY_COMPL_TOKEN_PREFIX + str(i))
            self.add_to_vocab(constants.tokeniser_constants.DURATION_COMPL_TOKEN_PREFIX + str(i))
            self.add_to_vocab(constants.tokeniser_constants.INTERVAL_COMPL_TOKEN_PREFIX + str(i))


    def add_bars_to_vocab(self) -> None:
        for i in range(1, self.config.additional_params[constants.tokeniser_constants.MAX_BARS_FIELD] + 1):
            self.add_to_vocab(constants.tokeniser_constants.BAR_TOKEN_PREFIX + str(i))


    def stuff_vocab_index_holes(self):
        for i, token in enumerate(self.vocab):
            self.vocab[token] = i


    def remove_unecessary_program_tokens_from_vocab(self):
        for token in self.vocab.copy():
            if token.startswith("Program"):
                if token in ["Program_0", "Program_2"]: 
                    continue
                del self.vocab[token]

    def encode_with_metadata(self, input_file: Path, tokenised_metadata: Metadata | Metadata.TokenisedMetadata | dict) -> dict:
        """
        Encode MIDI file with metadata conditioning tokens.
        
        Combines metadata tokens with musical content tokens to create complete
        input sequences for conditioned music generation models.
        
        Args:
            input_file: Path to MIDI file to encode
            tokenised_metadata: Metadata object or dictionary with conditioning info
            
        Returns:
            Dictionary containing:
            - metadata tokens: List of metadata token strings
            - tokens: List of all token strings (metadata + music)  
            - input_ids: List of all token IDs (metadata + music)
            - labels: List of music token IDs (for training targets)
            - tokenizer_hash: Hash for compatibility checking
        """
        metadata_seq = self.encode_metadata(tokenised_metadata)

        # encode_ids=true only encodes ids only if tokeniser is trained
        tok_seq = self.encode(input_file, encode_ids=True, no_preprocess_score=False, attribute_controls_indexes=None)

        return {constants.tokeniser_constants.TOKENS_METADATA_KEY: metadata_seq.tokens,
                constants.tokeniser_constants.TOKENS_TOKENS_KEY: tok_seq.tokens,
                constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY: metadata_seq.ids + tok_seq.ids,
                constants.tokeniser_constants.TOKENS_LABELS_KEY: tok_seq.ids,
                constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY: self.hexa_hash}

    def encode_metadata(self, tokenised_metadata: Metadata | Metadata.TokenisedMetadata | dict) -> miditok.TokSequence:
        """
        Encode metadata into token sequence format.
        
        Args:
            tokenised_metadata: Metadata in various formats to encode
            
        Returns:
            TokSequence with metadata tokens and IDs
        """
        tokenised_metadata = Metadata.TokenisedMetadata.type_to_dict(tokenised_metadata)
        
        metadata_seq = miditok.TokSequence(list(tokenised_metadata.values()))
        self.complete_sequence(metadata_seq, complete_bytes=True)
        if self.is_trained:
            self.encode_token_ids(metadata_seq)

        return metadata_seq

    def save_generated_tokens(self, output_file: Path, generated_tokens: list[int], key_signature: int, metadata: Metadata | Metadata.TokenisedMetadata | dict) -> None:
        """
        Save generated token sequences with metadata to JSON file.
        
        Args:
            output_file: Path where to save the token file
            generated_tokens: List of generated token IDs
            key_signature: Key signature as integer (sharps/flats)
            metadata: Metadata object or dictionary
        """
        import json

        if isinstance(metadata, Metadata):
            metadata = metadata.tokenised_metadata.as_dict
        elif isinstance(metadata, Metadata.TokenisedMetadata):
            metadata = metadata.as_dict

        with open(output_file, "w") as f:
            json.dump({
                constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY: generated_tokens,
                constants.tokeniser_constants.TOKENS_KEY_SIGNATURE_KEY: key_signature,
                constants.tokeniser_constants.TOKENS_METADATA_KEY: metadata,
                constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY: self.hexa_hash
            }, f, indent=4)

    def train_BPE(self, data_dir: Path):
        """
        Train Byte-Pair Encoding on MIDI files for vocabulary compression.
        
        This method builds a compressed vocabulary by learning frequent token
        pairs from the training data, reducing sequence lengths and improving
        model efficiency.
        
        Args:
            data_dir: Directory containing MIDI files for training
        """
        files = list(data_dir.glob(f"*{constants.MIDI_EXTENSION}"))

        print(f"Training BPE on {len(files)} {constants.MIDI_EXTENSION} files in {data_dir}")

        self.train(vocab_size=self.vocab_size * constants.tokeniser_constants.BPE_VOCAB_SCALE_FACTOR, model='BPE', iterator=miditok.tokenizer_training_iterator.TokTrainingIterator(self, files))

    @property
    def hexa_hash(self) -> str:
        """
        Generate SHA256 hash of tokenizer configuration for compatibility checking.
        
        The hash is based on the complete tokenizer dictionary representation
        and is used to verify that saved tokens were generated by a compatible
        tokenizer configuration.
        
        Returns:
            Hexadecimal string representation of configuration hash
            
        Note:
            May produce false negatives due to unsorted dictionary serialization,
            but guarantees that different configurations have different hashes.
        """
        import hashlib
        return hashlib.sha256(str(self.to_dict()).encode('utf-8')).hexdigest()

    def valid_metadata(self, tokenised_data: Metadata | Metadata.TokenisedMetadata | Dict[str, str]) -> Tuple[bool, str]:
        """
        Validate metadata against tokenizer constraints and vocabulary.
        
        Checks that metadata values are within acceptable ranges and that
        all metadata tokens exist in the tokenizer's vocabulary.
        
        Args:
            tokenised_data: Metadata to validate in various formats
            
        Returns:
            Tuple of (is_valid, error_message). If invalid, error_message
            contains detailed information about validation failures.
        """
        res = ""

        tokenised_data = Metadata.TokenisedMetadata.type_to_dict(tokenised_data)

        if (t := int(tokenised_data[constants.tokeniser_constants.NUM_MEASURES_FIELD].replace(constants.tokeniser_constants.BAR_TOKEN_PREFIX, ""))) > self.config.additional_params[constants.tokeniser_constants.MAX_BARS_FIELD]:
            res += f"Invalid number of measures: {t} not in range 1-{self.config.additional_params[constants.tokeniser_constants.MAX_BARS_FIELD]} Bars\n"

        if (t := tokenised_data[constants.tokeniser_constants.TIME_SIGNATURE_FIELD]) not in self.vocab:
            res += f"Invalid time signature: {t} not in {self.config.time_signature_range} Time Signatures\n"

        if res:
            return False, res
        
        return True, ""

    # those are overrides that ensure any tokeniser loading returns a MyTokeniser instance
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
        # note that super().from_pretrained() returns an instance of the class that it finds in the config
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

