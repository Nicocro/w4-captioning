import torch
import random
from PIL import Image
import matplotlib.pyplot as plt
import random
from dataset import Flickr30kDataset
from model import FlickrModel
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection
from typing import List

def load_model_from_checkpoint(checkpoint_path: str) -> FlickrModel:
    """Load the model from a checkpoint."""
    # Initialize the model components
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    
    # Initialize the model
    model = FlickrModel(
        clip_processor=clip_processor,
        clip=clip,
        clip_txt=clip_txt,
        num_dec_blocks=6,
        d_model=512,
        n_heads=1,
        ff_hidden_ratio=4
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def clean_caption(text: List) -> str:
    """Clean up caption text by removing special tokens and </w> tokens."""
    # If input is a list, join it first
    if isinstance(text, list):
        text = ' '.join(token for token in text 
                       if token not in ['<|startoftext|>', '<|endoftext|>'])
    
    # Remove </w> tokens and clean up spaces
    text = text.replace('</w>', '')
    text = ' '.join(text.split())
    
    return text

def display_images_and_captions(samples: list):
    """Display multiple images in a 2x3 grid with their captions."""
    fig = plt.figure(figsize=(20, 15))
    
    for idx, (image, gt_caption, gen_caption) in enumerate(samples, 1):
        plt.subplot(2, 3, idx)
        plt.imshow(image)
        plt.axis('off')
        
        # Create a two-line title with wrapped text
        title = f'Ground Truth:\n{gt_caption}\n\nGenerated:\n{gen_caption}'
        plt.title(title, wrap=True, size=10, pad=10)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    checkpoint_path = 'best_model_1.pth'  # Update this with your checkpoint path
    model = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Load validation dataset
    val_dataset = Flickr30kDataset(split='val')
    
    # Randomly select 6 images
    dataset_size = len(val_dataset)
    random_indices = random.sample(range(dataset_size), 6)
    
    # Store results
    results = []
    
    # Generate captions for selected images
    with torch.no_grad():
        for idx in random_indices:
            # Get an image and its caption from the dataset
            sample = val_dataset[idx]
            image = sample['image']
            ground_truth_caption = sample['caption']
            
            # Generate and clean caption
            generated_caption = model.generate_caption(image)
            clean_gen_caption = clean_caption(generated_caption)
            
            # Store results
            results.append((image, ground_truth_caption, clean_gen_caption))
    
    # Display all results in a grid
    display_images_and_captions(results)

if __name__ == '__main__':
    main()