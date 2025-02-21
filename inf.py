import torch
from PIL import Image
import matplotlib.pyplot as plt
from dataset import Flickr30kDataset
from model import FlickrModel
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection

def load_model_from_checkpoint(checkpoint_path):
    """Load the model from a checkpoint."""
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    
    model = FlickrModel(
        clip_processor=clip_processor,
        clip=clip,
        clip_txt=clip_txt,
        num_dec_blocks=8,
        d_model=512,
        n_heads=8,
        ff_hidden_ratio=4
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def clean_caption(text):
    """Clean up caption text by removing special tokens and </w> tokens."""
    if isinstance(text, list):
        text = ' '.join(token for token in text if token not in ['<|startoftext|>', '<|endoftext|>'])
    
    text = text.replace('</w>', '')
    text = ' '.join(text.split())
    
    return text

def display_image_and_captions(image, gt_caption, gen_caption):
    """Display a single image with ground truth and generated captions."""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    
    caption_text = f"Ground Truth:\n{gt_caption}\n\nGenerated:\n{gen_caption}"
    plt.title(caption_text, wrap=True, size=12, pad=10)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    checkpoint_path = 'best_model_large.pth'  # Update with your checkpoint path
    model = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Load validation dataset and get a single sample
    val_dataset = Flickr30kDataset(split='val')
    sample_idx = 43  # Can be changed to any valid index
    
    # Get image and caption
    sample = val_dataset[sample_idx]
    image = sample['image']
    ground_truth_caption = sample['caption']
    
    # Generate caption
    with torch.no_grad():
        generated_caption = model.generate_caption(image)
        clean_gen_caption = clean_caption(generated_caption)
    
    # Display results
    display_image_and_captions(image, ground_truth_caption, clean_gen_caption)

if __name__ == '__main__':
    main()