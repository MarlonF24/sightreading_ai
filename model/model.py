from transformers import GPT2Config, GPT2LMHeadModel
from pathlib import Path
from tokeniser.tokeniser import MyTokeniser, Metadata
from model.dataloader import MyTokenDataset
from typing import *
import constants as constants


class MyModel(GPT2LMHeadModel):
    """
    Custom GPT-2 language model for music generation with metadata conditioning.
    
    This model extends Hugging Face's GPT2LMHeadModel with specialized functionality
    for generating piano music sequences. It integrates with MyTokeniser to handle
    musical tokens and metadata conditioning, providing controllable music generation
    based on complexity and structural parameters.
    
    Key features:
    - Custom configuration building with tokenizer integration
    - Automatic sequence length analysis from training data
    - Smart model loading/creation with tokenizer compatibility checking
    - Metadata-conditioned generation for controllable output
    - Training pipeline integration with custom dataset handling
    
    Class Attributes:
        OWN_PATH: Path to this module file
        OWN_DIR: Directory containing this module
        TRAINING_DIR: Directory for saving/loading trained models
        OUTPUT_DIR: Directory for generated outputs
        
    Note:
        Requires MyTokeniser for proper token handling and metadata conditioning.
        Training data should be preprocessed into JSON token files.
        
    Example:
        >>> tokenizer = MyTokeniser()
        >>> model, loaded = MyModel.load_or_create(tokenizer, tokens_dir)
        >>> MyModel.train_from_tokens_dir(tokens_dir, tokenizer)
    """
    
    OWN_PATH = Path(__file__)
    OWN_DIR = OWN_PATH.parent
    TRAINING_DIR = OWN_DIR / constants.model_constants.TRAINING_DIR_NAME
    OUTPUT_DIR = OWN_DIR / constants.model_constants.OUTPUT_DIR_NAME

    @staticmethod
    def build_config(max_sequence_length: int, tokeniser: MyTokeniser) -> GPT2Config:
        """
        Build GPT-2 configuration with tokenizer-specific parameters.
        
        Creates a model configuration that includes tokenizer vocabulary size,
        special token IDs, and sequence length constraints derived from the
        provided tokenizer and training data analysis.
        
        Args:
            max_sequence_length: Maximum sequence length the model can handle
            tokeniser: MyTokeniser instance to extract vocabulary parameters from
            
        Returns:
            GPT2Config object configured for music generation with the given tokenizer
        """
        config = constants.model_constants.MYMODEL_BASE_CONFIG.copy()
        config[constants.model_constants.TOKENISER_HASH_FIELD] = tokeniser.hexa_hash
        config[constants.model_constants.VOCAB_SIZE_FIELD] = len(tokeniser.vocab_model) if tokeniser.is_trained else len(tokeniser.vocab)
        config[constants.model_constants.BOS_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.bos_token]
        config[constants.model_constants.EOS_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.eos_token]
        config[constants.model_constants.PAD_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.pad_token]
        config[constants.model_constants.MAX_POSITION_EMBEDDINGS_FIELD] = max_sequence_length

        return GPT2Config(**config)

    def __init__(self, config: GPT2Config):
        """
        Initialize MyModel with the given configuration.
        
        Args:
            config: GPT2Config object with model parameters
            
        Raises:
            ValueError: If config.architectures doesn't match this class name
        """
        if not config.architectures == [self.__class__.__name__]:
            raise ValueError(f"Expected config.architectures to be {self.__class__.__name__}, got {config.architectures}")

        super().__init__(config)

    @classmethod
    def load_or_create(cls, tokeniser: MyTokeniser, tokens_dir: Path) -> Tuple["MyModel", bool]:
        """
        Load existing model or create new one.
        
        This method checks for existing model files and validates tokenizer compatibility.
        If a compatible model exists, it's loaded. Otherwise, a new model is created
        with optimal configuration based on the training data.
        
        Args:
            tokeniser: MyTokeniser instance for compatibility checking
            tokens_dir: Directory containing training token files for analysis
            
        Returns:
            Tuple of (model_instance, was_loaded_from_disk)
            
        Raises:
            RuntimeError: If user aborts during tokenizer mismatch resolution
            
        Note:
            - Performs automatic sequence length analysis for new models
            - Validates tokenizer hash compatibility for loaded models
            - Provides interactive prompts for handling mismatches
        """
        import shutil, miditok.constants, transformers.utils

        files_present = [cls.TRAINING_DIR.joinpath(transformers.utils.CONFIG_NAME).exists(),
                cls.TRAINING_DIR.joinpath(transformers.utils.SAFE_WEIGHTS_NAME).exists(),
                cls.TRAINING_DIR.joinpath(transformers.utils.GENERATION_CONFIG_NAME).exists(),
                cls.TRAINING_DIR.joinpath(miditok.constants.DEFAULT_TOKENIZER_FILE_NAME).exists()]
        
        if all(files_present):
            print("All required files for loading found, loading model...")
            loaded_tokeniser = MyTokeniser.from_pretrained(pretrained_model_name_or_path=cls.TRAINING_DIR)

            loaded_tokeniser = MyTokeniser.from_pretrained(pretrained_model_name_or_path=cls.TRAINING_DIR)

            if loaded_tokeniser.hexa_hash == tokeniser.hexa_hash:
                print(f"✅ Tokeniser hash matches checkpoint model: {loaded_tokeniser.hexa_hash}")
                load = True
            else:
                print(f"⚠️ Tokeniser hash mismatch. Expected: {tokeniser.hexa_hash}, Found: {loaded_tokeniser.hexa_hash}")

                match input("Tokeniser hash does not match checkpoint model (I.e. So far the model was trained on tokens from a differently configured tokeniser). Choose:\n"
                            "[n] Create new model with given tokeniser\n"
                            "[a] Abort training\n> ").strip().lower():
                    case "n":
                        load = False
                    case _:
                        raise RuntimeError("Training aborted.")
    
        elif any(files_present):
            print("⚠️ Some, but not all files for loading are present.")
            if not input("Do you want to reinitialise the training directory with a new model for the given tokeniser? [y/n] ").strip().lower() == "y":
                raise RuntimeError("Training aborted.")
            load = False
        
        else:
            print("No files for loading found.")
            load = False

        if load:
            print("Loading model with given tokeniser...\n")
            return cls.from_pretrained(cls.TRAINING_DIR), True
        else:
            print("Initialising new model with given tokeniser.")
            shutil.rmtree(cls.TRAINING_DIR, ignore_errors=True)

            cls.TRAINING_DIR.mkdir(parents=True, exist_ok=False)

            max_sequence_length = cls.analyse_sequence_lengths_for_cutoff(tokens_dir, constants.model_constants.SEQUENCE_LENGTH_CUTOFF_PERCENTILE)

            return cls(cls.build_config(max_sequence_length + 2, tokeniser)), False  # +2 for BOS/EOS tokens

    @staticmethod
    def analyse_sequence_lengths_for_cutoff(tokens_dir: Path, target_percentile: float) -> int:
        """
        Analyze sequence lengths in token files and determine optimal cutoff.
        
        This method examines all JSON token files in a directory, extracts sequence
        lengths, and provides statistical analysis with visualization to help determine
        the optimal maximum sequence length for training.
        
        Args:
            tokens_dir: Directory containing JSON files with "input_ids" key
            target_percentile: Percentile (0-1) of files to keep based on length
            
        Returns:
            Maximum sequence length at the specified percentile cutoff
            
        Raises:
            ValueError: If target_percentile is not between 0 and 1
            
        Side Effects:
            - Prints histogram visualization (if plotext available)
            - Shows cutoff analysis for multiple percentiles
            - Displays memory usage estimates
            
        Note:
            Installing 'plotext' enables terminal histogram visualization.
            Memory estimates assume int32 tokens and batch size of 4.
        """
        import json
        
        if target_percentile < 0 or target_percentile > 1:
            raise ValueError("target_percentile must be between 0 and 1.")

        lengths = []
        for file in tokens_dir.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
            lengths.append(len(data["input_ids"])) 
        
        lengths.sort()
        total_files = len(lengths)
        
        # Calculate the selected max_length first
        target_idx = int(total_files * target_percentile)
        selected_max_length = lengths[target_idx - 1] if target_idx > 0 else max(lengths)
        
        # Create histogram visualization
        try:
            import plotext as plt
            
            plt.clear_data()
            plt.hist(lengths, bins=min(30, len(set(lengths))))  # Adaptive bins
            plt.title("🎵 Sequence Length Distribution")
            plt.xlabel("Sequence Length (tokens)")
            plt.ylabel("Number of Files")
            
            # Add vertical line for cutoff
            plt.vline(selected_max_length, color='red')
            
            # Set reasonable plot size for terminal
            plt.plotsize(80, 20)  # width, height in characters
            plt.show()
            
            print()  # Add spacing after plot
            
        except ImportError:
            print("📊 Install 'plotext' for histogram visualization: pip install plotext")
            print(f"📈 Data summary: {total_files} files, lengths from {min(lengths)} to {max(lengths)} tokens\n")
        
        # Show analysis for multiple cutoffs
        cutoffs = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995]
        
        print("📋 Cutoff Analysis:")
        print("─" * 70)
        for cutoff in cutoffs:
            idx = int(total_files * cutoff)
            kept_files = idx
            discarded_files = total_files - kept_files
            max_length_at_cutoff = lengths[idx - 1] if idx > 0 else 0
            
            # Highlight the target percentile
            marker = " ← 🎯 SELECTED" if cutoff == target_percentile else ""
            
            print(f"  {cutoff*100:4.1f}%: max_length={max_length_at_cutoff:4d}, "
                  f"keep {kept_files:4d} files, discard {discarded_files:2d} files{marker}")

        print("\n✅ Selected Configuration:")
        print(f"  📏 Max sequence length: {selected_max_length} tokens")
        print(f"  📁 Files kept: {target_idx:,}/{total_files:,} ({target_percentile*100:.1f}%)")
        print(f"  🗑️  Files discarded: {total_files - target_idx:,}")
        
        # Memory estimate in KB
        memory_per_token = 4  # bytes for int32
        est_memory_kb = (selected_max_length * memory_per_token * 4) / 1024  # batch_size=4
        print(f"  💾 Est. memory per batch: ~{est_memory_kb:.1f} KB")
        
        return selected_max_length

    # deprecated: gotta fix before reusing
    @classmethod
    def find_optimal_batch_size_for_model(cls, model: "MyModel", target_memory_mb: int = 1200) -> int:
        """
        Find optimal batch size for the given model and memory constraints.
        
        **DEPRECATED**: Needs fixing before reuse.
        
        Estimates memory usage for different batch sizes and finds the largest
        batch size that fits within the target memory limit.
        
        Args:
            model: The loaded model to analyze
            target_memory_mb: Target memory usage in MB (default 1200 for RTX 3050 Ti)
            
        Returns:
            Optimal batch size that fits within memory constraints
            
        Note:
            Memory estimates include model parameters, gradients, optimizer states,
            and activation memory for forward/backward passes.
        """
        # Get values directly from model config
        sequence_length = model.config.n_positions
        vocab_size = model.config.vocab_size
        n_embd = model.config.n_embd
        n_layer = model.config.n_layer

        # Find largest batch size that fits
        for batch_size in range(1, 33):
            # Quick memory estimate
            model_params_mb = (vocab_size * n_embd * 4 + sequence_length * n_embd * 4 + n_embd * n_embd * 4 * 12 * n_layer) / (1024**2)
            activations_mb = (batch_size * sequence_length * n_embd * 4 * 4 + batch_size * sequence_length * sequence_length * 4) / (1024**2)
            total_mb = model_params_mb * 4 + activations_mb  # 4x for params + gradients + optimizer
            
            if total_mb > target_memory_mb:
                optimal = max(1, batch_size - 1)
                print(f"🎯 Optimal batch size: {optimal}")
                return optimal
        
        print(f"🎯 Optimal batch size: 32 (max tested)")
        return 32

    @classmethod  
    def train_from_tokens_dir(cls, tokens_dir: Path, tokeniser: MyTokeniser):
        """
        Train model on tokenized music files using Hugging Face Trainer.
        
        This method handles the complete training pipeline: model loading/creation,
        dataset preparation, GPU setup, and training execution. It uses the custom
        MyTokenDataset for data loading and miditok's DataCollator for batching.
        
        Args:
            tokens_dir: Directory containing JSON token files for training
            tokeniser: MyTokeniser instance identical to the one used for tokenisation (hash based check)
            
        Side Effects:
            - Creates/updates model in TRAINING_DIR
            - Saves model and tokenizer after training
            - Prints GPU information and training progress
            - Creates training logs in the specified directory
            
        Note:
            - Automatically detects and uses GPU if available
            - Uses epoch-based saving strategy with total limit of 3 checkpoints
            - Batch size is currently fixed at 2 (optimal batch size method needs fixing)
        """
        import torch
        from transformers import Trainer, TrainingArguments
        
        cls.TRAINING_DIR.mkdir(parents=True, exist_ok=True)

        # Load or create model
        model, loaded = cls.load_or_create(tokeniser, tokens_dir=tokens_dir)
        
        # Move model to GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            print(f"✅ Model moved to: {device}")
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) # convert to GB
            print(f"📱 GPU Memory: {gpu_memory_gb:.1f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU (CUDA not available)")

        # Create dataset
        dataset = MyTokenDataset(
            files_paths=list(tokens_dir.glob(f"*{constants.TOKENS_EXTENSION}")),
            tokeniser=tokeniser,
            max_sequence_length=model.config.n_positions, 
            bos_token_id=tokeniser.vocab[tokeniser.bos_token],
            eos_token_id=tokeniser.vocab[tokeniser.eos_token],
            pad_token_id=tokeniser.vocab[tokeniser.pad_token]
        )

        from miditok.pytorch_data import DataCollator

        collator = DataCollator(
            copy_inputs_as_labels=False,
            shift_labels=False,
            pad_on_left=False,
            inputs_kwarg_name=constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY,
            labels_kwarg_name=constants.tokeniser_constants.TOKENS_LABELS_KEY,
            pad_token_id=tokeniser.vocab[tokeniser.pad_token],
            labels_pad_idx=-100
        )
        
        training_args = TrainingArguments(
            output_dir=str(cls.TRAINING_DIR),
            auto_find_batch_size=True,  
            save_strategy="epoch",
            logging_dir=str(cls.TRAINING_DIR / constants.model_constants.LOGS_DIR_NAME),
            save_total_limit=3,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

        trainer.train()
        trainer.model.save_pretrained(cls.TRAINING_DIR)
        if not loaded:
            tokeniser.save_pretrained(cls.TRAINING_DIR)

    

    @classmethod
    def generate_tokens(cls, metadata_tokens: Metadata.TokenisedMetadata, key_signature: int, output_dir: Path):
        """
        Generate music token sequences conditioned on metadata and key signature.
        
        Loads trained model and generates new music sequences based on the provided
        metadata conditioning tokens (complexity, structure) and target key signature.
        The generated tokens are saved as JSON files for further processing.
        
        Args:
            metadata_tokens: Tokenized metadata for conditioning generation
            key_signature: Target key signature (-7 to 7 sharps/flats)
            output_dir: Directory where generated token files will be saved
            
        Raises:
            ValueError: If key_signature is outside valid range (-7 to 7)
            ValueError: If metadata tokens are invalid for the loaded tokenizer
            
        Note:
            - Requires trained model and tokenizer in TRAINING_DIR
            - Uses greedy decoding by default (custom generation config commented)
            - Generated sequences include BOS token and metadata conditioning
            - Output files contain tokens, IDs, key signature, and metadata
        """
        if key_signature > 7 or key_signature < -7:
            raise ValueError(f"Invalid key signature: {key_signature}. Valid range is -7 to 7 sharps.")

        import torch
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokeniser and model
        tokeniser = MyTokeniser.from_pretrained(cls.TRAINING_DIR)

        t, err = tokeniser.valid_metadata(metadata_tokens)
        if not t:
            raise ValueError(f"Invalid metadata tokens: {err}")

        model = cls.from_pretrained(cls.TRAINING_DIR)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        tok_seq = tokeniser.encode_metadata(metadata_tokens)

        input_ids = torch.tensor([[tokeniser.vocab[tokeniser.bos_token]] + tok_seq.ids], dtype=torch.long).to(device)


        # custom_gen_config = model.generation_config
        # custom_gen_config.update(
        #     do_sample=True,
        #     temperature=0.9,
        #     top_k=50,
        #     top_p=0.95,
        #     repetition_penalty=1.2,
        #     max_new_tokens=128
        # )


        # Generate
        with torch.no_grad():
            generated = model.generate(input_ids)


        # Decode back to tokens
        output_ids = generated[0].tolist()

        tokeniser.save_generated_tokens(output_dir / f"generated{constants.TOKENS_EXTENSION}", output_ids, key_signature, metadata_tokens)


if __name__ == "__main__":
    tokens_dir = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens") 
    
    MyModel.train_from_tokens_dir(tokens_dir, MyTokeniser())