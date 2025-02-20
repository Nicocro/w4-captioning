import torch
from model import FiclkrModel
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection

def test_generate_caption():
    # Initialize the model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    
    model = FiclkrModel(
        clip_processor=clip_processor,
        clip=clip,
        clip_txt=clip_txt,
        num_dec_blocks=6,
        d_model=512,
        n_heads=1,
        ff_hidden_ratio=4
        )

    # Create a dummy image tensor with values in the range [0, 1]
    dummy_image = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.float32)  # Batch size 1, 3 color channels, 224x224 image size

    # Generate a caption
    generated_caption = model.generate_caption(dummy_image)

    print("Generated caption:", generated_caption)

    # Check if the generated caption starts with the
if __name__ == "__main__":
    test_generate_caption()
    print("All tests passed!")
