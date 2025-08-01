from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from mlm_dataset import load_mlm_dataset

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")

dataset = load_mlm_dataset("data/asm_corpus.txt", tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./pretrained_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=500,
    prediction_loss_only=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./pretrained_model")
tokenizer.save_pretrained("./pretrained_model")
