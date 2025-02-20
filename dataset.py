
from PIL import Image
import torch
from transformers import CLIPTokenizer, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List


class Flickr30kDataset(Dataset):
    def __init__(self, split: str='train'):
        super().__init__() 
        self.dataset = load_dataset("nlphuji/flickr30k", split='test') #type: ignore
        self.dataset = self.dataset.filter(lambda x: x['split'] == split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["caption"][0]

        return {"image": image, "caption": caption}


def collate_fn(batch: List[Dict], clip_tok: CLIPTokenizer, clip_proc: CLIPProcessor) -> Dict[str, torch.Tensor]:
    captions, images = zip(*[(item["caption"], item["image"]) for item in batch])

    tokenized = clip_tok(captions, truncation=True, padding=True, return_tensors="pt") # type: ignore
    tokenized_input = tokenized['input_ids'][:, :-1]  # type: ignore # remove [EOS] token
    attention_mask = tokenized["attention_mask"][:, :-1]  # Update attention mask # type: ignore
    tokenized_output = tokenized['input_ids'][:, 1:]  # remove [BOS] token # type: ignore

    with torch.no_grad():
        images = clip_proc(images=images, return_tensors="pt")["pixel_values"]

    return {
        "input_ids": tokenized_input,
        "attention_mask": attention_mask,
        "images": images, #type: ignore
        "output_ids": tokenized_output  # remove [BOS] token
    }


### MAIN TO TEST###

def main():

    dataset = Flickr30kDataset()
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(dataset[0])

    test_batch = DataLoader(dataset, batch_size=2, collate_fn=(lambda batch: collate_fn(batch, clip_tok, clip_proc))) #type: ignore

    
    for batch in test_batch:
        print(batch)

if __name__ == "__main__":
    main()