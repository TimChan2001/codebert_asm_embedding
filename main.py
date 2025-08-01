import os
from data.preprocess import preprocess_asm_function
from models.codebert_embedder import CodeBERTEmbedder
import numpy as np

def load_functions(path):
    with open(path, "r") as f:
        content = f.read()
    return [func.strip() for func in content.strip().split("\n\n") if func.strip()]

def main():
    embedder = CodeBERTEmbedder()
    functions = load_functions("data/asm_functions.txt")
    os.makedirs("embeddings", exist_ok=True)
    
    for i, func in enumerate(functions):
        processed = preprocess_asm_function(func)
        vec = embedder.get_embedding(processed, method="cls")
        np.save(f"embeddings/func_{i}.npy", vec)

if __name__ == "__main__":
    main()
