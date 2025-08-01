from datasets import load_dataset

def load_mlm_dataset(file_path, tokenizer, max_length=128):
    dataset = load_dataset("text", data_files={"train": file_path})
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])
