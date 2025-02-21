#imports 
from tracemalloc import start
import torch
from torch import nn
from transformers import CLIPModel, CLIPTextModelWithProjection, CLIPProcessor

from model_blocks import DecoderBlock
from typing import Dict, Tuple, Optional, List

class FlickrModel(nn.Module):
    def __init__(self,
                 clip_processor: CLIPProcessor,
                 clip = CLIPModel,
                 clip_txt = CLIPTextModelWithProjection,
                 num_dec_blocks: int=6,    
                 d_model: int=512,
                 n_heads: int=1,
                 ff_hidden_ratio: int=4,
                 ):
        super().__init__()
        self.clip_processor = clip_processor
        self.clip = clip
        self.clip_txt = clip_txt
        self.vocab = clip_processor.tokenizer.get_vocab() #get vocab from clip_processor #type: ignore
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab) #get vocab size from clip_processor
        self.max_len = 77 #max length of caption hardcoded for Compatibility with CLIP model 
        self.d_model = d_model

        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_len+1, d_model)) #initialize positional encoding
        self.blocks = nn.ModuleList([
            DecoderBlock(
                decoder_d_model=d_model,
                n_heads=n_heads,
                ff_hidden_ratio=ff_hidden_ratio)
            for _ in range(num_dec_blocks)
        ])
        self.out_proj = nn.Linear(d_model, self.vocab_size)

    def _embed_caption(self, example: torch.Tensor, attention_mask: torch.Tensor, clip_txt: CLIPTextModelWithProjection) -> torch.Tensor:
        """embed a caption using a CLIP text processor"""
        with torch.no_grad():
            clip_txt = clip_txt.to(example.device)
            outputs = clip_txt(input_ids=example, attention_mask=attention_mask) #type: ignore
            embedded_caption =  outputs.last_hidden_state 
        
        return embedded_caption

    def _embed_image(self, image: torch.Tensor, clip: CLIPModel) -> torch.Tensor:
        """embed an image using a CLIP image processor"""
        with torch.no_grad():  
            clip = clip.to(image.device)
            embedded_image = clip.get_image_features(pixel_values=image) #type: ignore
        
        return embedded_image

    
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        device = next(self.parameters()).device

        attention_mask = x['attention_mask'].to(device)
        image_embed = self._embed_image(x['images'], self.clip).unsqueeze(1).to(device) #embed image #type: ignore
        caption_embed = self._embed_caption(x['input_ids'], x['attention_mask'], self.clip_txt).to(device) #type: ignore

        input_seq = torch.cat([image_embed, caption_embed], dim=1) #concatenate image and caption embeddings
        image_attention = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)  # Extend attention mask to account for image token
        extended_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        input_seq = input_seq + self.positional_encoding[:, :input_seq.size(1), :].to(device) #slice positional encoding to match input_seq size
        for block in self.blocks:
            input_seq, attention = block(input_seq, extended_attention_mask)

        output = self.out_proj(input_seq)
        
        return output, attention #type: ignore
    
    
    def generate_caption(self, image: torch.Tensor) -> List:
        """Generate a caption for an image in an autoregressive manner"""
        start_token = self.clip_processor.tokenizer.bos_token #type: ignore
        end_token = self.clip_processor.tokenizer.eos_token #type: ignore
        start_token_id = self.vocab[start_token]
        end_token_id = self.vocab[end_token]
        
        self.eval()
        device = next(self.parameters()).device  # Get model's device

        with torch.no_grad():
            embedded_image = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            caption = [start_token_id]

            temperature = 0.4

            for _ in range(self.max_len):
                text_tensor = torch.tensor([caption], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(text_tensor)
                input_dict = {"input_ids": text_tensor, "attention_mask": attention_mask, "images": embedded_image}
                output, _ = self(input_dict)

                scaled_logits = output[0, -1, :] / temperature #apply temperature in generation
                probs = torch.softmax(scaled_logits, dim=0)
                next_token_id = torch.multinomial(probs, 1).item()

                caption.append(next_token_id)
                
                if next_token_id == end_token_id:
                    break
                
        caption = [self.inverse_vocab[token_id] for token_id in caption]

        return caption #return generated caption