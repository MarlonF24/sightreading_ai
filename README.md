# Sightreading AI

**A research project developing model-generated piano sight-reading exercises with controllable complexity using transformer models and symbolic music processing.**

## What This Project Does

This system shall eventually generate piano sight-reading exercises where users can specify what kind of musical challenge they want:

Users select musical challenge parameters using interactive controls (e.g., sliders, dropdowns, or checkboxes) for:

- **Number of bars** (e.g., 8, 16, 32)
- **Time signature** (e.g., 4/4, 3/4, 6/8)
- **Note density** (low, medium, high)
- **Rhythmic complexity** (simple, moderate, complex)
- **Melodic interval size** (small, medium, large)
- **Key signature** (e.g., C major, G major, F minor)

The model uses these metadata selections to condition the generation process, producing sight-reading exercises that match the chosen complexity and musical features.

## Technical Approach

### Pipeline Architecture
```
Musical Input → Complexity Analysis → Token Sequences → GPT-2 training → Controllable Generation
```

## Project Structure
```
sightreading_ai/
├── data_pipeline_scripts/     # Multi-format conversion pipeline
├── tokeniser/                 # Custom REMI tokenizer with metadata
├── model/                     # GPT-2 with conditioning, training scripts
├── constants/                 # Configuration management
└── model/training/            # Trained model checkpoints
```

