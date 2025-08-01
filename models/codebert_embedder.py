from transformers import RobertaTokenizer, RobertaModel
import torch

class CodeBERTEmbedder:
    def __init__(self):
        # self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        # self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("./pretrained_model")
        self.model = RobertaModel.from_pretrained("./pretrained_model")

        self.model.eval()

    def get_embedding(self, code: str, method="cls"):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if method == "cls":
            return outputs.last_hidden_state[0, 0, :].numpy()
        elif method == "mean":
            emb = outputs.last_hidden_state[0]
            mask = inputs["attention_mask"][0].unsqueeze(-1)
            return ((emb * mask).sum(0) / mask.sum(0)).numpy()
        else:
            raise ValueError("Unknown method")
