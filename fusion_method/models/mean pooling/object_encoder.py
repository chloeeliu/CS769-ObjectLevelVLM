import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoImageProcessor, AutoModel
from torch.nn.utils.rnn import pad_sequence

class ObjectEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        hid_dim_mapping = {
            "facebook/dino-vitb8": 768,
            "openai/clip-vit-large-patch14": 1024,
            "facebook/dinov2-large": 1024,
            "openai/clip-vit-large-patch14-336": 1024
        }
        hid_dim = hid_dim_mapping[self.config["vision_encoder"]]

        # Initialize the pre-trained vision encoder
        if "clip" in self.config["vision_encoder"]:
            self.model = CLIPVisionModel.from_pretrained(config["vision_encoder"]).to(config["device"])
            self.processor = CLIPImageProcessor.from_pretrained(self.config["vision_encoder"])
        elif "dino" in self.config["vision_encoder"]:
            self.model = AutoModel.from_pretrained('facebook/dinov2-large').to(config["device"])
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

        # Set requires_grad based on configuration
        self.model.requires_grad_(not config["freeze_vision_encoder"])

        # Define the projector to map embeddings to LLM space
        if self.config["llm_model"] == "meta-llama/Llama-2-7b-chat-hf":
            self.projector = nn.Linear(hid_dim, 4096)
        elif self.config["llm_model"] == "gpt2":
            self.projector = nn.Linear(hid_dim, 768)
        elif self.config["llm_model"] == "llava-hf/llava-1.5-7b-hf":
            self.projector = nn.Linear(hid_dim, 4096)

    def forward(self, segmentations, image_features):
        # Compute lengths of valid tokens per sample
        lengths = torch.sum(segmentations, axis=1).to(self.config["device"])
        max_length = max(lengths)
        # Create attention mask to handle variable sequence lengths
        attention_mask = torch.arange(max_length).to(self.config["device"])[None, :] < lengths[:, None]

        # Select the features corresponding to the segmentation masks
        selected_features = [image_features[i][segmentations[i]] for i in range(image_features.shape[0])]
        # Pad sequences to have consistent length across the batch
        features = pad_sequence(selected_features, batch_first=True)

        # Compute mean pooling over the valid tokens
        attention_mask = attention_mask.to(features.dtype)
        attention_mask = attention_mask.unsqueeze(-1)  # Reshape to [batch_size, seq_len, 1]
        sum_embeddings = torch.sum(features * attention_mask, dim=1)
        sum_mask = attention_mask.sum(dim=1)  # Sum of attention_mask to get the number of valid tokens
        sum_mask = sum_mask.masked_fill(sum_mask == 0, 1)  # Avoid division by zero
        pooled_output = sum_embeddings / sum_mask

        # Project the pooled output to the LLM embedding space
        return self.projector(pooled_output)
