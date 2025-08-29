# Sightreading-AI

Generate sight-reading exercises using AI.

## Goals
- Train a transformer model to generate short melodic piano exercises based on metadata input (tempo, key, complexity, etc.).
   1. Collect data: pdf, mxl, musicxml, midi
   2. Convert to symbolic music formats: musicxml/midi
   3. Load and tokenise symbolic music formats, extract metadata
   4. Train model on midi tokens and their metadata
   5. Convert generated midi tokens back to sheet music formats: musicxml, PDF (or web)

- Optional: Add real-time MIDI feedback for interactive learning.

## Project Structure

### 1. Data Conversion Pipeline

The pipeline is managed by the `Pipeline` and `Converter` classes in `data_pipeline_scripts/pipeline.py` and `data_pipeline_scripts/converter.py`. Each stage is represented by a `PipelineStage` object, and conversion functions are implemented as subclasses of `SingleFileConversionFunction` or `BatchConversionFunction` (see `conversion_func_infrastructure.py` and `conversion_functions.py`).

#### Pipeline Stages:
- **PDF → MXL**: Uses Audiveris via `pdf_to_mxl` (batch conversion).
- **MXL → MusicXML**: Via `mxl_to_musicxml_music21` (using music21) or `mxl_to_musicxml_unzip` (unzipping).
- **MusicXML → MIDI**: Via `to_midi` (using music21, also extracts and saves metadata).
- **MIDI → Tokens**: Via `midi_to_tokens` (using miditok, inserts metadata tokens).
- **Tokens**: Used for model training.

#### ConversionOutcome
Each conversion step returns a `ConversionOutcome` object, which tracks input/output files, warnings, errors, and skip/halt status.

#### Logging
All conversions are logged via the `Log` class, which tracks statistics and writes detailed logs for each conversion route.

### 2. Tokenisation and Metadata

- Tokenisation is handled by `tokenisation.py`, using the `miditok` library and a custom REMI tokenizer.
- Metadata is extracted from scores using the `Metadata` class, which computes key signature, time signature, clefs, pitch range, number of measures, and complexity metrics.
- Metadata tokens are inserted into the token sequence for each MIDI file.

### 3. Model

- The model is defined in `model/model.py` using HuggingFace's GPT2 architecture.
- The vocabulary is built from the tokenizer, and special tokens (BOS, EOS, PAD) are set.
- Training is managed via HuggingFace's `Trainer` API.

## Quick Start

1. **Create a Virtual Environment**:  
   Set up a new virtual environment to manage dependencies.

2. **Install Dependencies**:  
   Use the `requirements.txt` file to install necessary packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Pipeline**:  
   Configure paths for MuseScore and Audiveris in `pipeline.py` if needed.

4. **Run Conversion**:  
   Use `converter.py` to run multi-stage or single-stage conversions. Example:
   ```python
   converter = Converter(pipeline)
   converter.multi_stage_conversion(converter.pipeline["pdf_in"], converter.pipeline["tokens"], overwrite=True, batch_if_possible=True)
   ```

5. **Train Model**:  
   Prepare token datasets and use `model/model.py` to train the transformer model.

## Main Classes and Files

- **conversion_func_infrastructure.py**:  
  Defines the abstract base classes for conversion functions and the `ConversionOutcome` dataclass.

- **conversion_functions.py**:  
  Implements all conversion functions for each pipeline stage.

- **tokenisation.py**:  
  Handles tokenisation, metadata extraction, and custom vocab additions.

- **pipeline.py**:  
  Defines the pipeline structure and stages.

- **converter.py**:  
  Manages the conversion process, logging, and orchestration.

- **model/model.py**:  
  Defines and trains the transformer model.

## Additional Information

- **ConversionOutcome Classification**:  
  The project includes a classification system for `ConversionOutcome` objects, helping to understand the success, warnings, and errors encountered during the conversion process. ![ConversionOutcome Classification](stuff/image.png)

- **Logging**:  
  A comprehensive logging system tracks data processing and conversion stages.

- **Extensibility**:  
  The pipeline and conversion functions are designed to be extensible for new formats or processing steps.

---

*For more details, see the docstrings in each Python file.*