from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, Trainer, TrainingArguments
from pathlib import Path
from tokeniser.tokeniser import MyTokeniser
from model.dataloader import MyTokenDataset
from typing import *


class MyModel(GPT2LMHeadModel):
    OWN_PATH = Path(__file__)
    OWN_DIR = OWN_PATH.parent
    TRAINING_DIR = OWN_DIR / "training"
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    

    @staticmethod
    def build_config(tokeniser: MyTokeniser) -> GPT2Config:
        return GPT2Config(
            vocab_size=len(tokeniser.vocab_model),
            n_embd=512,
            n_layer=6,
            n_head=8,
            bos_token_id=tokeniser.vocab[tokeniser.bos_token],
            eos_token_id=tokeniser.vocab[tokeniser.eos_token],
            pad_token_id=tokeniser.vocab[tokeniser.pad_token],
            tokeniser_hash=tokeniser.hexa_hash,
        )

    @classmethod
    def load_or_create(cls, tokeniser: MyTokeniser) -> Tuple["MyModel", bool]:
        import json, shutil
        from transformers import trainer_utils

        if trainer_utils.get_last_checkpoint(str(cls.TRAINING_DIR)):
            print("Checkpoint found, loading model...")

            loaded_tokeniser = MyTokeniser.from_pretrained(pretrained_model_name_or_path=cls.TRAINING_DIR)

            if loaded_tokeniser.hexa_hash == tokeniser.hexa_hash:
                print("✅ Tokeniser hash matches. Continuing with existing model.")
                load = True

            else:
                print(f"⚠️ Tokeniser hash mismatch. Expected: {tokeniser.hexa_hash}, Found: {loaded_tokeniser.hexa_hash}")

                match input("Tokenizer hash does not match checkpoint model. Choose:\n"
                            "[n] Create new model with new tokeniser\n"
                            "[a] Abort training\n> ").strip().lower():
                    case "n":
                        print("Creating new model with new tokeniser.")
                        load = False
                    case _:
                        raise RuntimeError("Training aborted.")
        else:
            print("No checkpoint found. Initialising new model.")
            load = False

        if load:
            return cls.from_pretrained(cls.TRAINING_DIR), True
        else:
            shutil.rmtree(cls.TRAINING_DIR, ignore_errors=True)
            cls.TRAINING_DIR.mkdir(parents=True, exist_ok=False)
            return cls(MyModel.build_config(tokeniser)), False

    @classmethod
    def train_from_tokens_folder(cls, tokens_folder: Path, tokeniser: MyTokeniser):
        from transformers import Trainer, TrainingArguments, trainer_utils

        model, loaded = cls.load_or_create(tokeniser)
        dataset = MyTokenDataset(
            files_paths=list(tokens_folder.glob("*.json")),
            tokeniser_hash=tokeniser.hexa_hash,
            bos_token_id=tokeniser.vocab[tokeniser.bos_token],
            eos_token_id=tokeniser.vocab[tokeniser.eos_token],
            pad_token_id=tokeniser.vocab[tokeniser.pad_token]
        )

        from miditok.pytorch_data import DataCollator

        collator = DataCollator(
            copy_inputs_as_labels=False,
            shift_labels=False,
            pad_on_left=False,
            inputs_kwarg_name="input_ids",
            labels_kwarg_name="labels",
            pad_token_id=tokeniser.vocab[tokeniser.pad_token],
            labels_pad_idx=-100
        )
        

        training_args = TrainingArguments(
            output_dir=str(cls.TRAINING_DIR),
            per_device_train_batch_size=2,
            save_strategy="epoch",
            logging_dir=str(cls.TRAINING_DIR / "logs"),
            save_total_limit=3,  # Optional: keep only last 3 checkpoints
            resume_from_checkpoint=trainer_utils.get_last_checkpoint(str(cls.TRAINING_DIR))
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

if __name__ == "__main__":
    tokens_folder = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens") 
    
    MyModel.train_from_tokens_folder(tokens_folder, MyTokeniser())