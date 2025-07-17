from importlib.metadata import metadata
import music21, miditok
from typing import *
from dataclasses import dataclass
from functools import cached_property
import music21.stream.base


TIME_SIGNATURES = {8: [3, 12, 6, 9], 4: [5, 6, 3, 2, 1, 4]}
CLEFS = ['G', 'F']
MAX_BARS = 34
PITCH_RANGE = (21, 108)  # MIDI note numbers for piano range (A0 to C8)

TOKENISER_CONFIG = miditok.classes.TokenizerConfig(use_programs=True, 
                                            use_time_signatures=True,
                                            pitch_range=PITCH_RANGE,
                                            #use_chords=True,
                                            use_rests=True,
                                            time_signature_range=TIME_SIGNATURES,
                                            #chord_tokens_with_root_note=True,
                                            #chord_unknown=(2, 4),
                                            one_token_stream_for_programs=True)
    

tokeniser = miditok.REMI(tokenizer_config=TOKENISER_CONFIG, max_bar_embedding=MAX_BARS)
original_encode = tokeniser.encode
    
def custom_encode(*args, **kwargs):
    tok_seq = original_encode(*args, **kwargs)
    tok_seq.tokens = ['BOS_None'] + tok_seq.tokens + ['EOS_None']
    tok_seq.ids = [tokeniser.vocab["BOS_None"]] + tok_seq.ids + [tokeniser.vocab["EOS_None"]]
    return tok_seq

def add_key_signatures_to_vocab(tokeniser) -> None:
    l = len(tokeniser._vocab_base)
    for i in range(-7, 8):
        tokeniser._vocab_base[f'KeySig_{i}'] = l + 7 + i

def add_clefs_to_vocab(tokeniser) -> None:
    l = len(tokeniser._vocab_base)
    for i in range(0, 2):
        tokeniser._vocab_base[f'Clef_{CLEFS[i]}'] = l + i

def add_complexities_to_vocab(tokeniser) -> None:
    l = len(tokeniser._vocab_base)
    for i in range(1, 11):
        tokeniser._vocab_base[f'Dens_{i}'] = l + i
        tokeniser._vocab_base[f'Dur_{i}'] = l + 10 + i
        tokeniser._vocab_base[f'Int_{i}'] = l + 20 + i

tokeniser.encode = custom_encode
add_key_signatures_to_vocab(tokeniser)
add_clefs_to_vocab(tokeniser)
add_complexities_to_vocab(tokeniser)

# print(tokeniser._vocab_base)


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

    @staticmethod
    def valid_metadata(tokenised_data: Dict[str, str]) -> Tuple[bool, str]:
        if tokenised_data["num_measures"] not in tokeniser._vocab_base:
            return False, f"Invalid number of measures: {tokenised_data['num_measures']} not in range 1-{MAX_BARS} Bars"
        
        if (t := tokenised_data["time_signature"]) not in tokeniser._vocab_base:
            return False, f"Invalid time signature: {t} not in {TIME_SIGNATURES}"

        if (c := tokenised_data["rh_clef"]) not in tokeniser._vocab_base:
            return False, f"Invalid RH clef: {c} not in {CLEFS}"

        if (c := tokenised_data["lh_clef"]) not in tokeniser._vocab_base:
            return False, f"Invalid LH clef: {c} not in {CLEFS}"

        if (l := tokenised_data["lowest_pitch"]) not in tokeniser._vocab_base:
            return False, f"Invalid lowest pitch: {l} not in MIDI range {PITCH_RANGE}"

        if (h := tokenised_data["highest_pitch"]) not in tokeniser._vocab_base:
            return False, f"Invalid highest pitch: {h} not in MIDI range {PITCH_RANGE}"
        
        return True, ""



if __name__ == "__main__":
    pass