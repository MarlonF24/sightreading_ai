from transformers import GPT2Config, GPT2LMHeadModel
from data_pipeline_scripts.tokenisation import tokeniser

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

tokeniser.encode()
model = GPT2LMHeadModel(config)

def train_model(train_dataset, val_dataset, epochs=3, batch_size=32, learning_rate=5e-5):
    from transformers import Trainer, TrainingArguments
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()