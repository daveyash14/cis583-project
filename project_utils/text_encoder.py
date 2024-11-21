from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def encode(self, captions):
        tokens = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        embeddings = self.model(**tokens).last_hidden_state
        return embeddings.mean(dim=1)