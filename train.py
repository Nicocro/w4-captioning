import torch
from torch import nn
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPTextModelWithProjection, CLIPProcessor

#my dependencies
from model_blocks import DecoderBlock
from dataset import Flickr30kDataset, collate_fn
from model import FlickrModel

#logging and progress visualization
import tqdm
import wandb

# type hints
from typing import Dict, Tuple, Optional, List

## TRAINING SCRIPT ## 

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm.tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        output, _ = model(batch)
        output = output[:, 1:, :]  # Remove predictions for the image token
        target = batch["output_ids"]
        
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = target.reshape(-1)
        
        loss = criterion(output_flat, target_flat)
        
        loss = criterion(output_flat, target_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        wandb.log({"batch_tr_loss": loss.item()})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            output, _ = model(batch)
            output = output[:, 1:, :] # Remove predictions for the image token
            target = batch["output_ids"]
            
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    #wandb logging
    wandb.init(project="flickr30k-captioning")  

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 5
    lr = 1e-4
    
    # Initialize CLIP components
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Initialize model
    model = FlickrModel(
        clip_processor=clip_processor,
        clip=clip,
        clip_txt=clip_txt,
        num_dec_blocks=6,
        d_model=512,
        n_heads=4,
        ff_hidden_ratio=4
    ).to(device)
    
    # Initialize datasets and dataloaders
    train_dataset = Flickr30kDataset(split='train')
    val_dataset = Flickr30kDataset(split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, clip_processor.tokenizer, clip_processor)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, clip_processor.tokenizer, clip_processor)
    )
    
    # Initialize optimizer and criterion
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print("Saved best model checkpoint")

            artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{epoch}", 
                    type="model"
                )
            
            artifact.add_file("model.pth")

    wandb.finish()

if __name__ == "__main__":
    main()