import torch
from PIL import Image
import matplotlib.pyplot as plt
from dataset import Flickr30kDataset
from model import FlickrModel
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection

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
        n_heads=4,
        ff_hidden_ratio=4
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def display_image_and_caption(image: Image.Image, caption: str, generated_caption: list):
    """Display the image and both the ground truth and generated captions."""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Join the generated caption tokens and remove special tokens
    generated_text = ' '.join(token for token in generated_caption 
                            if token not in ['<|startoftext|>', '<|endoftext|>'])
    
    plt.title(f'Ground Truth: {caption}\nGenerated: {generated_text}', 
              wrap=True, pad=20)
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    checkpoint_path = 'best_model.pth'  # Update this with your checkpoint path
    model = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Load validation dataset
    val_dataset = Flickr30kDataset(split='val')
    
    # Number of images to test
    num_test_images = 5
    
    # Generate captions for some validation images
    with torch.no_grad():
        for i in range(num_test_images):
            # Get an image and its caption from the dataset
            sample = val_dataset[i]
            image = sample['image']
            ground_truth_caption = sample['caption']
            
            # Generate caption
            generated_caption = model.generate_caption(image)
            
            # Display results
            display_image_and_caption(image, ground_truth_caption, generated_caption)
            
            # Print separation line
            print('-' * 80)

if __name__ == '__main__':
    main()
