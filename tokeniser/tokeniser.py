import music21, miditok
from typing import *
from functools import cached_property, wraps
from pathlib import Path

class Metadata:
    #weights for complexity measures
    DENSITY_COMPLEXITY_WEIGHT = 1
    DURATION_COMPLEXITY_WEIGHT = 1
    INTERVAL_COMPLEXITY_WEIGHT = 1
    
    
    def __init__(self, score: music21.stream.base.Score):
        self._score: music21.stream.base.Score = score
        if len(self._score.parts) != 2:
            raise ValueError("Expected two staves (RH and LH), got something else.")
        
    
    @property 
    def score(self) -> music21.stream.base.Score:
        return self._score

    @score.setter
    def score(self, score: music21.stream.base.Score):
        raise PermissionError("Score cannot be modified after initialization.")

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
    def key_signatures(self) -> List[int]:
        signatures = self.rh_part.recurse().getElementsByClass(music21.key.KeySignature)
        return [signature.sharps for signature in signatures] if signatures else [0]  # 0 = C Major/A minor
    
    @cached_property
    def time_signatures(self) -> List[str]:
        signatures = self.rh_part.recurse().getElementsByClass(music21.meter.TimeSignature)
        return [signature.ratioString for signature in signatures] if signatures else ['4/4']
    
    @cached_property
    def num_measures(self) -> int:
        return len(self.rh_part.getElementsByClass(music21.stream.Measure))
    
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
    def density_complexity(self) -> float:
        note_density = Metadata.DENSITY_COMPLEXITY_WEIGHT * round(self.num_total_notes / max(self.num_measures, 1) / 2)
        return min(10, max(1, note_density)) # Complexity (placeholder heuristic: density)
        
    @cached_property
    def duration_complexity(self) -> float:
        unique_durations =  set(round(n.quarterLength, 2) for n in self.notes)
        return min(10, max(1, Metadata.DURATION_COMPLEXITY_WEIGHT * len(unique_durations)))  # Duration complexity (how many types of durations are used?)
    
    @cached_property
    def interval_complexity(self) -> float:
        avg_interval = Metadata.INTERVAL_COMPLEXITY_WEIGHT * sum(self.intervals) / len(self.intervals) if self.intervals else 0.0
        return min(10, round(avg_interval / 2))  # Pitch complexity (based on interval size variability)

    @cached_property
    def data(self) -> Dict[str, Any]:
        return {
            "key_signature": self.key_signatures[0],
            "time_signature": self.time_signatures[0],
            "clefs": {"RH": self.rh_clefs[0], "LH": self.lh_clefs[0]},
            "pitch_range": self.pitch_range,
            "num_measures": self.num_measures,
            "density_complexity": self.density_complexity,
            "duration_complexity": self.duration_complexity,
            "pitch_complexity": self.interval_complexity
        }


    @cached_property
    def tokenised_data(self) -> Dict[str, str]:
        return {
            "key_signature": f"KeySig_{self.key_signatures[0]}",
            "time_signature": f"TimeSig_{self.time_signatures[0]}",
            "rh_clef": f"Clef_{self.rh_clefs[0]}",
            "lh_clef": f"Clef_{self.lh_clefs[0]}",
            "lowest_pitch": f"Pitch_{self.pitch_range[0].midi}",
            "highest_pitch": f"Pitch_{self.pitch_range[1].midi}",
            "num_measures": f"Bar_{self.num_measures}",
            "density_complexity": f"Dens_{self.density_complexity}",
            "duration_complexity": f"Dur_{self.duration_complexity}",
            "interval_complexity": f"Int_{self.interval_complexity}"
        }
    
class MyTokeniserConfig(miditok.classes.TokenizerConfig):
        """
        Custom configuration for the MyTokeniser.
        This class inherits from miditok.classes.TokenizerConfig and can be extended if needed.
        """
        import miditok.constants
        
        CONFIG = {
            "use_programs": True,
            "use_time_signatures": True,
            # "use_chords": True,
            "use_rests": True,
            # "chord_tokens_with_root_note": True,
            # "chord_unknown": (2, 4),
            "one_token_stream_for_programs": True,
            "special_tokens": miditok.constants.SPECIAL_TOKENS # dont change
        }
        
        def __init__(self, 
                     time_signature_range: dict[int, list[int]] = {8: [3, 12, 6, 9], 4: [5, 6, 3, 2, 1, 4]}, 
                     clefs: list[str] = ['G', 'F'],
                     pitch_range: tuple[int, int] = (21, 108),
                     max_bars: int = 33):
            """
            Initialize the MyTokeniserConfig with custom parameters.
            :param time_signature_range: Dictionary defining valid time signatures.
            :param clefs: List of valid clefs.
            :param pitch_range: Tuple defining the MIDI pitch range.
            :param max_bars: Maximum number of bars for embedding.
            """
            super().__init__(**self.CONFIG,
                             pitch_range=pitch_range,
                             clefs=clefs,
                             time_signature_range=time_signature_range,
                             max_bar_embedding=max_bars)
                   

class MyTokeniser(miditok.REMI):
    """
    Custom tokeniser class that extends miditok.REMI.
    This class is used to build a tokeniser for MIDI files using the REMI scheme.
    """ 
    
    def __init__(self, tokenizer_config: MyTokeniserConfig = MyTokeniserConfig(), params: Optional[Path] = None):
        
        super().__init__(tokenizer_config=tokenizer_config, params=params)
        
        self.bos_token = 'BOS_None'
        self.eos_token = 'EOS_None'
        self.pad_token = 'PAD_None'

        # trained tokenisers should have their whole vocab initialised again,
        # untrained ones get theirs created again, without our metadata tokens        

        if not self.is_trained:
            self.add_key_signatures_to_vocab()
            self.add_clefs_to_vocab()
            self.add_complexities_to_vocab()


    def encode_with_metadata(self, input_file: Path, tokenised_metadata: dict) -> dict:
        metadata_seq = miditok.TokSequence(list(tokenised_metadata.values()))
        self.complete_sequence(metadata_seq, complete_bytes=True)
        self.encode_token_ids(metadata_seq)

        tok_seq = self.encode(input_file, encode_ids=True, no_preprocess_score=False, attribute_controls_indexes=None)

        
        return {"input_ids": metadata_seq.ids + tok_seq.ids, "labels": tok_seq.ids, "tokeniser_hash": self.hexa_hash}


    def train_BPE(self, data_dir: Path):
        """
        Train the tokeniser on a dir of MIDI files.
        This method will build the vocabulary based on the MIDI files in the specified dir.
        """
        # from data_pipeline_scripts.conversion_functions import Generics

        self.train(vocab_size=self.vocab_size * 2, model='BPE', iterator=miditok.tokenizer_training_iterator.TokTrainingIterator(self, list(data_dir.glob("*.midi"))))
        
        # Generics.clear_n_terminal_lines(2)
    

    def add_key_signatures_to_vocab(self) -> None:
        for i in range(-7, 8):
            self.add_to_vocab(f'KeySig_{i}')


    def add_clefs_to_vocab(self) -> None:
        for clef in self.config.additional_params["clefs"]:
            self.add_to_vocab(f'Clef_{clef}')


    def add_complexities_to_vocab(self) -> None:
        for i in range(1, 11):
            self.add_to_vocab(f'Dens_{i}')
            self.add_to_vocab(f'Dur_{i}')
            self.add_to_vocab(f'Int_{i}')


    @property
    def hexa_hash(self) -> str:
        """
        Hash is based on unsorted tokeniser json as returned by to_dict() !!! allows for false negatives !!!
        """
        
        import hashlib

        return hashlib.sha256(str(self.to_dict()).encode('utf-8')).hexdigest()


    def valid_metadata(self, tokenised_data: Metadata) -> Tuple[bool, str]:
        res = ""
        
        if tokenised_data["num_measures"] not in self.vocab:
            res += f"Invalid number of measures: {tokenised_data['num_measures']} not in range 1-{self.config.additional_params['max_bar_embedding']} Bars\n"

        if (t := tokenised_data["time_signature"]) not in self.vocab:
            res += f"Invalid time signature: {t} not in {self.config.time_signature_range} Time Signatures\n"

        if (c := tokenised_data["rh_clef"]) not in self.vocab:
            res += f"Invalid RH clef: {c} not in {self.config.additional_params['clefs']} Clefs\n"

        if (c := tokenised_data["lh_clef"]) not in self.vocab:
            res += f"Invalid LH clef: {c} not in {self.config.additional_params['clefs']} Clefs\n"

        if (l := tokenised_data["lowest_pitch"]) not in self.vocab:
            res += f"Invalid lowest pitch: {l} not in MIDI range {self.config.pitch_range}\n"

        if (h := tokenised_data["highest_pitch"]) not in self.vocab:
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

