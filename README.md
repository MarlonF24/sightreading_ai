# Sightreading AI

**A machine learning pipeline for generating piano sight-reading exercises using transformer models and symbolic music processing.**

## Overview

This project (still WOP, see current development areas later) implements a system for training a machine learning model that shall eventually generate piano sight-reading exercises with control over musical complexity and structure. This combines optical music recognition (OMR), symbolic music processing, custom tokenisation, and transformer-based generation.

### Key Features

- **Flexible Data Pipeline**: Multi-stage conversion supporting PDF or direct MIDI input
- **Custom Tokeniser**: Metadata-conditioned REMI tokeniser with piano-specific optimisations
- **Complexity Metrics**: Automated assessment of complexity metrics (e.g. density, rhythmic, and melodic complexity)
- **Controllable Generation**: Generate exercises with specific complexity, structure, and key signatures

## Architecture

### 1. Data Processing Pipeline For Model Training

The pipeline transforms raw musical data through multiple stages with error handling and logging:

**From PDF sources:**
```
PDF → MXL → MIDI → Tokens → Model Training → Generation
```

**From MIDI sources:**
```
MIDI → MIDI(processed) → Tokens → Model Training → Generation
```

#### Pipeline Components

- **`Pipeline`** and **`Converter`** classes orchestrate the conversion process
- **`PipelineStage`** objects define each transformation step
- **Conversion Functions** implement specific format transformations with flexible processing modes:
  - **`SingleFileConversionFunction`**: Processes files individually with granular error tracking
  - **`BatchConversionFunction`**: Supports both single-file and batch processing for efficiency
  - Format implementations:
    - `pdf_to_mxl`: Audiveris OMR for PDF → MXL conversion (batch-capable for parallel processing)
    - `to_midi`: music21-based conversion with metadata extraction (single-file for detailed analysis)
    - `midi_to_tokens`: Custom tokenisation with metadata conditioning (batch-capable for efficiency)

#### Error Handling & Logging

- **`ConversionOutcome`** objects track success, warnings, and errors for each file
- **`Log`** class provides comprehensive statistics and detailed conversion logs
- **Performance optimisation**: Skipping existing files and efficient resource management

#### Additional Features

- **Input Validation**: File filtering according to musical constraints implied by the tokeniser config 
- **Score Refurbishing**: Integrated option to refurbish scores (add missing time/key sig., split into separate exercises, correct errors by Audiveris)
- **PDF-Preprocessing**: split pages to facilitate OMR -> risk of cutting individual exercises into pieces (potentially add resolution enhancement, brightness/contrast adjustment; see https://audiveris.github.io/audiveris/_pages/guides/advanced/improved_input/)
- **Temporary file saving system**: Move files out of the pipeline to separate directories according to outcome


### 2. Tokenisation & Metadata System

#### Custom Tokeniser (`MyTokeniser`)

Extends miditok's REMI tokeniser with:
- **Metadata conditioning tokens** (complexity, structure, key signatures)
- **Piano-optimised vocabulary** (removes drums, unused instruments)
- **BPE training support** to enhance semantical value of vocab
- **Hash-based compatibility checking** to ensure that a model only receives data from one tokeniser configuration

#### Metadata Extraction (`Metadata` class)

Computes musical metrics:
- **Density Complexity**: Note density relative to measure count
- **Duration Complexity**: Rhythmic variety based on note duration diversity
- **Interval Complexity**: Melodic complexity from average interval sizes
- **Structural Information**: Time signatures, key signatures, measure counts, etc. 

### 3. Model Architecture

#### Custom GPT-2 Implementation (`MyModel`)

- **Metadata-Conditioned Generation**: Uses extracted complexity tokens as conditioning
- **Smart Loading**: Handles model/tokeniser compatibility and version management
- **Training Pipeline**: Integrated with HuggingFace Trainer for training
- **Sequence Length Analysis**: Optimal cutoff determination from training data



## Installation & Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Java Runtime Environment (for Audiveris OMR, if using PDF input)
- [Audiveris OMR software](https://audiveris.github.io/audiveris/) (if using PDF input)

### Installation

1. **Clone Repository**:
    ```bash
    git clone https://github.com/MarlonF24/sightreading_ai
    cd sightreading_ai
    ```

2. **Setup Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate sightreading_ai
   ```

   Or alternatively:
   ```bash
   conda env create --prefix .conda -f environment.yml
   conda activate ./.conda
   ```

3. **Configure External Tools** (if using PDF input):
   - Install Audiveris and note the installation path
   - Install languages (English, German) in Audiveris under Tools > Install languages...
   - Update paths in `data_pipeline_constants.py` if needed

## Usage

### Quick Start: Complete Training Pipeline

```python
from pathlib import Path
from tokeniser.tokeniser import MyTokeniser
from model.model import MyModel
from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter

# 1. Setup tokeniser
tokeniser = MyTokeniser()

# 2. Setup pipeline
pipeline = construct_music_pipeline(tokeniser=tokeniser)
converter = Converter(pipeline=pipeline)

# 3. Convert to MIDI stage (from PDF or place MIDI files directly in midi_start/)
# Option A: From PDF files
converter.multi_stage_conversion("pdf_in", "midi_in", batch_if_possible=False, overwrite=True, move_successful_inputs_to_temp=True, move_error_inputs_to_temp=True)

# Option B: Place MIDI files in data_pipeline/data/midi_start/

converter.multi_stage_conversion("midi_start", "midi_in", batch_if_possible=False, overwrite=True, move_successful_inputs_to_temp=True, move_error_inputs_to_temp=True)

# 4. Train BPE tokeniser (optional, for vocabulary compression)
tokeniser.train_BPE(data_dir=Path("data_pipeline/data/midi_in"))
tokeniser.save_pretrained("trained_tokeniser")

# 5. Convert MIDI to tokens
converter.multi_stage_conversion("midi_in", "tokens_in", batch_if_possible=False, overwrite=True)

# 6. Train model
MyModel.train_from_tokens_dir(
    tokens_dir=Path("data_pipeline/data/tokens_in"), 
    tokeniser=tokeniser
)
```


### Generating Music

```python
from tokeniser.tokeniser import Metadata
from model.model import MyModel

# Define musical parameters
metadata = Metadata.TokenisedMetadata(
    time_signature="4/4",
    num_measures=16,
    density_complexity=5,    # 1-10 scale
    duration_complexity=3,   # 1-10 scale  
    interval_complexity=4    # 1-10 scale
)

# Generate exercise
MyModel.generate_tokens(
    metadata_tokens=metadata,
    key_signature=0,  # C major/A minor
    output_dir=Path("generated_exercises")
)
```

## Project Structure

```
sightreading_ai/
├── data_pipeline_scripts/          # Data processing pipeline
│   ├── conversion_functions.py     # Format conversion implementations
│   ├── converter.py               # Pipeline orchestration
│   └── pipeline.py                # Pipeline configuration
├── tokeniser/                     # Custom tokenisation system
│   ├── tokeniser.py              # MyTokeniser implementation
│   └── tokeniser_constants.py    # Tokeniser configuration
├── model/                         # Model architecture and training
│   ├── model.py                  # MyModel implementation
│   ├── dataloader.py             # Custom PyTorch dataset
│   └── model_constants.py        # Model configuration
├── constants/                     # Global configuration
└── data_pipeline/                 # Data processing directories (generated by converter initialisation)
    ├── logs/                     # Conversion logs and statistics
    ├── temp/                     # Temporary files during processing
    ├── error_temp/               # Files that failed conversion (for debugging)
    └── data/
        ├── pdf_in/               # Input PDF files
        ├──...
        ├── midi_start/           # Input MIDI files 
        └── tokens_in/            # Tokenised sequences for training
    
```


## Current Develompent Areas
- Finding optimal tokeniser configuration
- Finding eligible and sizeable dataset to build prototype
- Finetune complexity metrics
 



