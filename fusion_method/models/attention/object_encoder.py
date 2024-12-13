import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import CLIPModel, CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig, AutoImageProcessor, AutoModel
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)



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
        if "clip" in self.config["vision_encoder"]:
            self.model = CLIPVisionModel.from_pretrained(config["vision_encoder"]).to(config["device"])
            self.processor = CLIPImageProcessor.from_pretrained(self.config["vision_encoder"])
        elif "dino" in self.config["vision_encoder"]:
            self.model = AutoModel.from_pretrained('facebook/dinov2-large').to(config["device"])
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            

        self.model.requires_grad_(not config["freeze_vision_encoder"])
        
        vision_config = CLIPVisionConfig(hidden_size=hid_dim, num_hidden_layers = 2, num_attention_heads = 8, patch_size = 14)
        self.transformer = CLIPVisionModel(vision_config)
        self.transformer.vision_model.forward = self.new_vision_forward

        if self.config["llm_model"] == "meta-llama/Llama-2-7b-chat-hf":
            self.projector = nn.Linear(hid_dim, 4096)
        elif self.config["llm_model"] == "gpt2":
            self.projector = nn.Linear(hid_dim, 768)
        elif self.config["llm_model"] == "llava-hf/llava-1.5-7b-hf":
            self.projector = nn.Linear(hid_dim, 4096)

        #define the attention layer
        self.attention_fc = nn.Linear(hid_dim, 1)

    def new_vision_forward(self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None):
         
        hidden_states = pixel_values

        if attention_mask is not None:
           attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.transformer.vision_model.encoder(
        inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.transformer.vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return last_hidden_state, pooled_output, hidden_states


    # Expects segmentations to be (b_s, patch_size ** 2 + 1) binary mask tensor
    # Returns (batch_size, hidden dimension tensor) (representation of object)
    # Which can be inserted directly into the embedding space of LLM
    def forward(self, segmentations, image_features):
        lengths = torch.sum(segmentations, axis = 1).to(self.config["device"])
        max_length = lengths.max().item()
        attention_mask = torch.arange(max_length).to(self.config["device"])[None, :] < lengths[:, None]
        #collect features corresponding to the object patches
        features_list = [
            image_features[i][segmentations[i]] for i in range(image_features.shape[0])
        ]
        features = pad_sequence(features_list, batch_first=True)
        # Compute attention scores
        attn_scores = self.attention_fc(features).squeeze(-1)
        # Mask the padding positions
        attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Compute weighted sum of features
        attn_weights = attn_weights.unsqueeze(-1)
        attended_features = (features * attn_weights).sum(dim=1)
        # attended_features shape: (batch_size, hidden_dim)

        #test = [image_features[i][segmentations[i]] for i in range(image_features.shape[0])]
        #features = pad_sequence(test, batch_first=True)

        if self.config["no_compression"]:
            return self.projector(attended_features) #features[0]
        else: # get pooled representation of image patches and project to LLM space
            out = self.transformer.vision_model(
                pixel_values=attended_features.unsqueeze(1)
            )[1]
            #out = self.transformer.vision_model(pixel_values = features, attention_mask = attention_mask)[1]
            return self.projector(out)
