# Sightreading-AI

Generate sight-reading exercises using AI.

## Goals
- Load and tokenize symbolic music (MIDI, MusicXML)
- Train a transformer model to generate short melodic exercises
- Export to sheet music (PDF or web)
- Optional: Add real-time MIDI feedback

## Project Structure
- Sheet music (MusicXML) → MIDI → Tokens → [train model]
- Generated tokens → MIDI → Sheet music (MusicXML or PDF)

## Quick Start
1. Create virtualenv
2. Install dependencies
3. Run sample tokenizer script

## To Do
- [ ] Add example MIDI files
- [ ] Write tokenizer to convert to tokens
- [ ] Try first model generation
