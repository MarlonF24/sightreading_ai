import music21
from typing import *
from dataclasses import dataclass
from functools import cached_property
import music21.stream.base

class Metadata:
    #weights for complexity measures
     
    def __init__(self, score: music21.stream.base.Score):
        self._score: music21.stream.base.Score = score
        
    
    @property 
    def score(self) -> music21.stream.base.Score:
        return self._score

    @score.setter
    def score(self, score: music21.stream.base.Score):
        raise PermissionError("Score cannot be modified after initialization.")

    @cached_property
    def part(self) -> music21.stream.base.Part:
        return self.score.parts[0]

    @cached_property
    def tempos(self) -> List[float]:
        tempos = self.part.recurse().getElementsByClass(music21.tempo.MetronomeMark)
        return [t for tempo in tempos if (t := tempo.getQuarterBPM())] if tempos else [80] 

    @cached_property
    def key_signatures(self) -> List[int]:
        signatures = self.part.recurse().getElementsByClass(music21.key.KeySignature)
        return [signature.sharps for signature in signatures] if signatures else [0]  # 0 = C Major/A minor
    
    @cached_property
    def time_signatures(self) -> List[str]:
        signatures = self.part.recurse().getElementsByClass(music21.meter.TimeSignature)
        return [signature.ratioString for signature in signatures] if signatures else ['4/4']
    
    @cached_property
    def num_measures(self) -> int:
        return len(self.part.getElementsByClass(music21.stream.Measure))
    
    @cached_property
    def clefs(self) -> List[str]:
        return [clef.sign for clef in self.part.getElementsByClass(music21.clef.Clef) if clef.sign]
    
    @cached_property
    def notes(self) -> List[music21.note.Note]:
        return self.part.flat.notes

    @cached_property
    def midi_values(self) -> List[int]:
        midi_values: List[int] = []
        for n in self.notes:
            if isinstance(n, music21.note.Note):
                midi_values.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                midi_values.extend(p.midi for p in n.pitches)  # all pitches in the chord
        return midi_values
    
    @cached_property
    def num_total_notes(self) -> int:
        return len(self.part.flat.notes)

    @cached_property
    def pitch_range(self) -> Tuple[str, str]:
        if self.midi_values:
            pitch_min = music21.pitch.Pitch(midi=min(self.midi_values)).nameWithOctave
            pitch_max = music21.pitch.Pitch(midi=max(self.midi_values)).nameWithOctave
            return (pitch_min, pitch_max)
        
        return ("C4", "C5")  # fallback

    @cached_property
    def intervals(self) -> List[int]:
        return [abs(n2 - n1) for n1, n2 in zip(self.midi_values[:-1], self.midi_values[1:])]
    
    @cached_property
    def density_complexity(self) -> float:
        note_density = round(self.num_total_notes / max(self.num_measures, 1) / 2)
        return min(10, max(1, note_density)) # Complexity (placeholder heuristic: density)
        
    @cached_property
    def duration_complexity(self) -> float:
        unique_durations = set(round(n.quarterLength, 2) for n in self.notes)
        return min(10, max(1, len(unique_durations)))  # Duration complexity (how many types of durations are used?)
    
    @cached_property
    def interval_complexity(self) -> float:
        avg_interval = sum(self.intervals) / len(self.intervals) if self.intervals else 0.0
        return min(10, round(avg_interval / 2))  # Pitch complexity (based on interval size variability)


    @cached_property
    def data(self) -> Dict[str, Any]:
        return {
            "tempo": round(self.tempos[0]),
            "key_signature": self.key_signatures[0],
            "time_signature": self.time_signatures[0],
            "clef": self.clefs[0:2],
            "pitch_range": self.pitch_range,
            "num_measures": self.num_measures,
            "density_complexity": self.density_complexity,
            "duration_complexity": self.duration_complexity,
            "pitch_complexity": self.interval_complexity
        }
    
