# Sightreading-AI

Generate sight-reading exercises using AI.

## Goals
- Load and tokenize symbolic music formats (MIDI, MusicXML).
- Train a transformer model to generate short melodic exercises.
- Export generated exercises to sheet music formats (PDF or web).
- Optional: Add real-time MIDI feedback for interactive learning.

## Project Structure
The project is organized to facilitate the conversion and processing of musical data through various stages:

1. **Data Conversion Pipeline**:
   - **Sheet Music (MusicXML) → MIDI → Tokens**: This pipeline processes input sheet music into a format suitable for model training.
   - **Generated Tokens → MIDI → Sheet Music (MusicXML or PDF)**: Converts model-generated tokens back into readable sheet music.

2. **Tools and Dependencies**:
   - **MuseScore**: Utilized for converting and handling music files.
   - **Audiveris**: Used for optical music recognition to convert PDF sheet music to MusicXML.

## Quick Start
To get started with the Sightreading-AI project, follow these steps:

1. **Create a Virtual Environment**: Set up a new virtual environment to manage dependencies.
2. **Install Dependencies**: Use the `requirements.txt` file to install necessary packages.
   ```bash
   pip install -r requirements.tx

## To Do
- [ ] Add example MIDI files to the project for testing and demonstration.
- [ ] Develop a tokenizer to convert music into tokens for model training.
- [ ] Test the initial model generation process to ensure it produces valid musical exercises.

## Additional Information
- **ConversionOutcome Classification**: The project includes a classification system for `ConversionOutcome` objects, which is visually represented in ![ConversionOutcome Classification](image.png). This classification helps in understanding the success, warnings, and errors encountered during the conversion process.
- **musescore.exe** and **audiveris.exe** are required for certain conversion processes.
- The project includes a comprehensive logging system to track data processing and conversion stages.