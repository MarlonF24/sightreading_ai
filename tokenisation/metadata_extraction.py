import music21
from typing import *

def extract_metadata(score: music21.stream.base.Score) -> Dict[str, Any]:
        """
        Extract relevant metadata from a music21 Score object for sight-reading generation.

        Returns:
            A dictionary with structured metadata fields.
        """
        part = score.parts[0]  # Assume single-part sight-reading exercises

        # Tempo
        tempos = part.metronomeMarkBoundaries()
        tempo = tempos[0][2].getQuarterBPM() if tempos else 80  # Default fallback

        # Key Signature
        ks = part.getElementsByClass(music21.key.KeySignature).first()
        key_signature = ks.sharps if ks else 0  # 0 = C Major/A minor

        # Time Signature
        ts = part.getTimeSignatures()[0] if part.getTimeSignatures() else music21.meter.TimeSignature('4/4')
        time_signature = ts.ratioString

        # Clef
        clef_obj = part.getElementsByClass(music21.clef.Clef).first()
        clef_type = clef_obj.sign if clef_obj else 'treble'

        # Pitch range
        notes = part.recurse().notes
        midi_values: List[int] = []

        for n in notes:
            if isinstance(n, music21.note.Note):
                midi_values.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                midi_values.extend(p.midi for p in n.pitches)  # all pitches in the chord

        if midi_values:
            pitch_min = music21.pitch.Pitch(midi=min(midi_values)).nameWithOctave
            pitch_max = music21.pitch.Pitch(midi=max(midi_values)).nameWithOctave
            pitch_range = (pitch_min, pitch_max)
        else:
            pitch_range = ("C4", "C5")  # fallback

        # Complexity (placeholder heuristic: density)
        total_notes = len(midi_values)
        num_measures = len(part.getElementsByClass(music21.stream.Measure))
        note_density = total_notes / max(num_measures, 1)

        complexity = min(10, max(1, round(note_density / 2)))  # crude heuristic

        # Duration complexity (how many types of durations are used?)
        unique_durations = set(round(n.quarterLength, 2) for n in notes)
        duration_complexity = min(10, len(unique_durations))

        # Pitch complexity (based on interval size variability)
        intervals_list = [
            abs(n2 - n1)
            for n1, n2 in zip(midi_values[:-1], midi_values[1:])
        ]
        avg_interval = sum(intervals_list) / len(intervals_list) if intervals_list else 0.0
        pitch_complexity = min(10, round(avg_interval / 2))

        return {
            "tempo": round(tempo),
            "key_signature": key_signature,
            "time_signature": time_signature,
            "clef": clef_type,
            "pitch_range": pitch_range,
            "num_measures": num_measures,
            "complexity": complexity,
            "duration_complexity": duration_complexity,
            "pitch_complexity": pitch_complexity
        }