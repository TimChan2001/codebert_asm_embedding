# Assembly Code Embedding Scheme Based on CodeBERT

### Environment

- ./requirements.txt

### File Structure

```bash
codebert_asm_embedding/
├── pretrain/                         # Module for pretraining
│   ├── mlm_dataset.py                # Convert assembly text to MLM dataset
│   ├── run_pretrain.py               # Main training script using HuggingFace Trainer
│   └── train_config.json             # Optional: Training configuration file
├── data/
│   ├── asm_corpus.txt                # Large-scale assembly function corpus, one sample per line
│   ├── asm_functions.txt             # Small-scale data for embedding extraction
│   └── preprocess.py                 # Assembly code preprocessing script
├── models/
│   └── codebert_embedder.py          # CodeBERT embedding extractor
├── embeddings/                       # Output folder for generated .npy or .txt embeddings
├── main.py                           # Main entry point: batch embedding generation
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation

```



### Data Preparation

1. Write multiple assembly functions into the `data/asm_functions.txt` file, separating each function with a blank line:

```assembly
_start:
    mov eax, 1
    mov ebx, 0
    int 0x80

func1:
    push ebp
    mov ebp, esp
    xor eax, eax
    pop ebp
    ret
...
```

These are the assembly functions that need to be embedded.

1. Prepare the pretraining corpus of assembly code.

Collect **a large number of real-world assembly functions/instruction sequences**, from tools such as:

- Sources: `objdump`, `Ghidra`, `angr`, `Binary Ninja`, ELF disassembly data, etc.
- Format: Each function or code block should be represented as a single sentence

Example data (each line is an assembly function or code snippet):

```assembly
push ebp ; mov ebp , esp ; sub esp , 0x10 ; mov eax , [ebp+8] ; ...
mov eax , 1 ; mov ebx , 0 ; int 0x80
xor eax , eax ; ret
...
```

Save this as a `.txt` file with one sentence per line.

------

### Fine-tuning CodeBERT on Assembly Code (MLM)

```bash
python pretrain/run_pretrain.py
```

This script will fine-tune the CodeBERT model using the corpus in `data/asm_corpus.txt` via a **Masked Language Modeling (MLM)** task.

The fine-tuned model will be saved in:

```bash
./pretrained_model/
```

------

### Generating Embeddings

```bash
python main.py
```

After execution, a `.npy` embedding file will be generated for each function under the `embeddings/` directory:

```bash
embeddings/
├── func_0.npy
├── func_1.npy
...
```

