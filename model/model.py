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
    def build_config(tokeniser: MyTokeniser = MyTokeniser()) -> GPT2Config:
        config = constants.model_constants.MYMODEL_BASE_CONFIG.copy()
        config[constants.model_constants.TOKENISER_HASH_FIELD] = tokeniser.hexa_hash
        config[constants.model_constants.VOCAB_SIZE_FIELD] = len(tokeniser.vocab_model) if tokeniser.is_trained else len(tokeniser.vocab)
        config[constants.model_constants.BOS_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.bos_token]
        config[constants.model_constants.EOS_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.eos_token]
        config[constants.model_constants.PAD_TOKEN_ID_FIELD] = tokeniser.vocab[tokeniser.pad_token]

        return GPT2Config(**config)

    def __init__(self, config: GPT2Config):
        if not config.architectures == [self.__class__.__name__]:
            raise ValueError(f"Expected config.architectures to be {self.__class__.__name__}, got {config.architectures}")

        super().__init__(config)
    

    @classmethod
    def load_or_create(cls, tokeniser: MyTokeniser) -> Tuple["MyModel", bool]:
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
        else:
            load = False

        if load:
            print("Loading model with given tokeniser...")
            return cls.from_pretrained(cls.TRAINING_DIR), True
        else:
            if any(files_present):
                print("⚠️ Some, but not all files for loading are present.")
                if not input("Do you want to reinitialise the training directory with a new model for the given tokeniser? [y/n] ").strip().lower() == "y":
                    raise RuntimeError("Training aborted.")
            else:
                print("No files to load found.")

            print("Initialising new model with given tokeniser.")
            shutil.rmtree(cls.TRAINING_DIR, ignore_errors=True)
            cls.TRAINING_DIR.mkdir(parents=True, exist_ok=False)
            return cls(cls.build_config(tokeniser)), False

    @classmethod
    def train_from_tokens_dir(cls, tokens_dir: Path, tokeniser: MyTokeniser):
        from transformers import Trainer, TrainingArguments, trainer_utils
        cls.TRAINING_DIR.mkdir(parents=True, exist_ok=True)


        model, loaded = cls.load_or_create(tokeniser)
        dataset = MyTokenDataset(
            files_paths=list(tokens_dir.glob(f"*{constants.TOKENS_EXTENSION}")),
            tokeniser=tokeniser,
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
    def generate_tokens(cls,metadata_tokens: Metadata.TokenisedMetadata):
        import torch
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

        input_ids = torch.tensor([tok_seq.ids], dtype=torch.long).to(device)

        # Generate
        with torch.no_grad():
            generated = model.generate(input_ids, generation_config=model.generation_config)


        # Decode back to tokens
        output_ids = generated[0].tolist()

        tokeniser.save_generated_tokens(cls.OUTPUT_DIR / f"generated{constants.tokeniser_constants.TOKENS_EXTENSION}", output_ids, metadata_tokens)

if __name__ == "__main__":
    tokens_dir = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens") 
    
    MyModel.train_from_tokens_dir(tokens_dir, MyTokeniser())