from transformers import GPT2Config, GPT2LMHeadModel
from pathlib import Path
from tokeniser.tokeniser import MyTokeniser, Metadata
from model.dataloader import MyTokenDataset
from typing import *
import constants as constants


class MyModel(GPT2LMHeadModel):
    OWN_PATH = Path(__file__)
    OWN_DIR = OWN_PATH.parent
    TRAINING_DIR = OWN_DIR / constants.model_constants.TRAINING_DIR_NAME
    OUTPUT_DIR = OWN_DIR / constants.model_constants.OUTPUT_DIR_NAME

    @staticmethod
    def build_config(max_sequence_length: int, tokeniser: MyTokeniser = MyTokeniser()) -> GPT2Config:
        config = constants.model_constants.MYMODEL_BASE_CONFIG.copy()
        config[constants.model_constants.TOKENISER_HASH_FIELD] = tokeniser.hexa_hash
        config[constants.model_constants.VOCAB_SIZE_FIELD] = len(tokeniser.vocab_model) if tokeniser.is_trained else len(tokeniser.vocab)
        config[constants.model_constants.BOS_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.bos_token]
        config[constants.model_constants.EOS_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.eos_token]
        config[constants.model_constants.PAD_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.pad_token]
        config[constants.model_constants.MAX_POSITION_EMBEDDINGS_FIELD] = max_sequence_length

        return GPT2Config(**config)

    def __init__(self, config: GPT2Config):
        if not config.architectures == [self.__class__.__name__]:
            raise ValueError(f"Expected config.architectures to be {self.__class__.__name__}, got {config.architectures}")

        super().__init__(config)
    

    @classmethod
    def load_or_create(cls, tokeniser: MyTokeniser, tokens_dir: Path) -> Tuple["MyModel", bool]:
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

                match input("Tokenizer hash does not match checkpoint model. Choose:\n"
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
            print("Loading model with given tokeniser...")
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
        Analyzes the sequence lengths in JSON files within a directory and determines the maximum sequence length
        to keep a specified percentile of files.
        Args:
            tokens_dir (Path): Directory containing JSON files, each with an "input_ids" key.
            target_percentile (float): The percentile (between 0 and 1) of files to keep based on sequence length.
        Returns:
            int: The maximum sequence length at the specified percentile cutoff.
        Raises:
            ValueError: If target_percentile is not between 0 and 1.
        Side Effects:
            Prints an analysis of sequence length cutoffs for several percentiles, highlighting the selected cutoff.
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
        
        # Show analysis for multiple cutoffs
        cutoffs = [0.90, 0.95, 0.98, 0.99, 0.995]
        
        print("Cutoff analysis:")
        for cutoff in cutoffs:
            idx = int(total_files * cutoff)
            kept_files = idx
            discarded_files = total_files - kept_files
            max_length_at_cutoff = lengths[idx - 1] if idx > 0 else 0
            
            # Highlight the target percentile
            marker = " ← SELECTED" if cutoff == target_percentile else ""
            
            print(f"  {cutoff*100:4.1f}%: max_length={max_length_at_cutoff:4d}, "
                f"keep {kept_files:4d} files, discard {discarded_files:2d} files{marker}")
    
        # Calculate the selected max_length
        target_idx = int(total_files * target_percentile)
        selected_max_length = lengths[target_idx - 1] if target_idx > 0 else max(lengths)
        
        print(f"\nSelected {target_percentile*100}% cutoff:")
        print(f"  Max sequence length: {selected_max_length}")
        print(f"  Files kept: {target_idx}/{total_files}")
        print(f"  Files discarded: {total_files - target_idx}")
        
        return selected_max_length


    @classmethod
    def train_from_tokens_dir(cls, tokens_dir: Path, tokeniser: MyTokeniser):
        import torch
        from transformers import Trainer, TrainingArguments
        
        cls.TRAINING_DIR.mkdir(parents=True, exist_ok=True)


        model, loaded = cls.load_or_create(tokeniser, tokens_dir=tokens_dir)

       
        # MOVE MODEL TO GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            print(f"✅ Model moved to: {device}")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU (CUDA not available)")


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
            per_device_train_batch_size=2,
            save_strategy="epoch",
            logging_dir=str(cls.TRAINING_DIR / constants.model_constants.LOGS_DIR_NAME),
            save_total_limit=3,  # Optional: keep only last 3 checkpoints
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
    def generate_tokens(cls,metadata_tokens: Metadata.TokenisedMetadata, key_signature: int, output_dir: Path):
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


        custom_gen_config = model.generation_config
        custom_gen_config.update(
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_tokens=128
        )


        # Generate
        with torch.no_grad():
            generated = model.generate(input_ids, generation_config=custom_gen_config)


        # Decode back to tokens
        output_ids = generated[0].tolist()

        tokeniser.save_generated_tokens(output_dir / f"generated{constants.TOKENS_EXTENSION}", output_ids, key_signature, metadata_tokens)

if __name__ == "__main__":
    tokens_dir = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens") 
    
    MyModel.train_from_tokens_dir(tokens_dir, MyTokeniser())