from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, Trainer, TrainingArguments
from tokeniser.tokeniser import tokeniser
from pathlib import Path


vocab = tokeniser.vocab

config = GPT2Config(
    vocab_size=len(vocab),
    n_embd=512,
    n_layer=6,
    n_head=8,
    bos_token_id=vocab['BOS_None'],
    eos_token_id=vocab['EOS_None'],
    pad_token_id=vocab['PAD_None']
)


def combine_tokens(tokens_folder: Path, output_folder: Path) -> Path:
    import json
    output_folder.mkdir(parents=True, exist_ok=True)
    combined_tokens_path = output_folder / "combined_tokens.jsonl"
    with combined_tokens_path.open("w", encoding="utf-8") as out_file:
        for token_file in tokens_folder.glob("*.json"):
            with token_file.open("r", encoding="utf-8") as f:
                json.dump(json.load(f), out_file)
                out_file.write("\n")
    print(f"Combined tokens saved to {combined_tokens_path}")
    return combined_tokens_path


def train_from_tokens_folder(tokens_folder: Path, model_cls=AutoModelForCausalLM):
    from datasets import load_dataset
    
    training_folder = Path(__file__).parent / "training"
    training_folder.mkdir(parents=True, exist_ok=True)

    dataset_folder = training_folder / "dataset"
    dataset_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(__file__).parent / "checkpoint_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Combine all token files
    combined_tokens_path = combine_tokens(tokens_folder, dataset_folder)

    # Step 2: Load dataset
    dataset = load_dataset("json", data_files=str(combined_tokens_path), split="train")

    import os

    # Step 3: Load or initialize model
    if os.listdir(checkpoint_dir):
        model = model_cls.from_pretrained(checkpoint_dir)
    else:
        model = GPT2LMHeadModel(config)

    # Step 4: Training setup
    training_args = TrainingArguments(
        output_dir=training_folder,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_dir=f"{training_folder}/logs",
        resume_from_checkpoint=checkpoint_dir
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  
    )

    # Step 5: Train
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    # Step 6: Save final model
    model.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    tokens_folder = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens") 
    train_from_tokens_folder(tokens_folder, tokeniser)