from PIL import Image
from torch.utils.data import Dataset
import os

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Load captions
        self.image_caption_pairs = []
        with open(captions_file, "r") as f:
            for line in f.readlines():
                image_name, caption = line.strip().split('\t')
                self.image_caption_pairs.append((image_name.split("#")[0], caption))

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_name, caption = self.image_caption_pairs[idx]
        image_path = os.path.join(self.root_dir, "Images", image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        if self.tokenizer:
            tokens = self.tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")
            caption_embedding = tokens["input_ids"].squeeze(0)
        else:
            caption_embedding = caption

        return image, caption_embedding