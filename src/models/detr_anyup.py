# src/models/detr_anyup.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

class DetrWithAnyUpSwitch(nn.Module):
    def __init__(self, model_name_or_path: str, anyup_model: nn.Module):
        super().__init__()
        self.raw_detr = DetrForObjectDetection.from_pretrained(model_name_or_path)
        self.anyup = anyup_model
        self.use_anyup = False

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        pixel_mask: torch.Tensor | None = None, 
        labels: list[dict] | None = None
    ):
        if not self.use_anyup:
            return self.raw_detr(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            
        backbone_outputs = self.raw_detr.model.backbone(pixel_values, pixel_mask=pixel_mask)
        feature_map = backbone_outputs.last_hidden_state       
        mask = backbone_outputs.last_hidden_state_mask          
        
        upsampled_features = self.anyup(feature_map)
        
        if mask is not None:
            new_h, new_w = upsampled_features.shape[-2:]
            mask = nn.functional.interpolate(
                mask.float().unsqueeze(1), 
                size=(new_h, new_w), 
                mode="nearest"
            ).squeeze(1).bool()
            
        backbone_outputs.last_hidden_state = upsampled_features
        backbone_outputs.last_hidden_state_mask = mask
        
        transformer_outputs = self.raw_detr.model(
            pixel_values=None,           
            pixel_mask=None,
            backbone_outputs=backbone_outputs, 
            return_dict=True
        )
        
        logits = self.raw_detr.class_labels_classifier(transformer_outputs.last_hidden_state)
        pred_boxes = self.raw_detr.bbox_predictor(transformer_outputs.last_hidden_state)
        
        return {
            "logits": logits, 
            "pred_boxes": pred_boxes
        }