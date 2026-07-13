from __future__ import annotations
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

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
        feature_map = backbone_outputs.last_hidden_state      # [B, 2048, H, W]
        mask = backbone_outputs.last_hidden_state_mask          
        
        proj_features = self.raw_detr.model.input_projection(feature_map) # [B, 2048, H, W] -> [B, 256, H, W]
        upsampled_features = self.anyup(proj_features)                    # [B, 256, 2H, 2W]
        
        if mask is not None:
            new_h, new_w = upsampled_features.shape[-2:]
            mask = nn.functional.interpolate(
                mask.float().unsqueeze(1), 
                size=(new_h, new_w), 
                mode="nearest"
            ).squeeze(1).bool()
            
        orig_projection = self.raw_detr.model.input_projection
        self.raw_detr.model.input_projection = nn.Identity()
        
        backbone_outputs.last_hidden_state = upsampled_features
        backbone_outputs.last_hidden_state_mask = mask
        
        try:
            transformer_outputs = self.raw_detr.model(
                pixel_values=None,           
                pixel_mask=None,
                backbone_outputs=backbone_outputs, 
                return_dict=True
            )
        finally:
            self.raw_detr.model.input_projection = orig_projection
        
        logits = self.raw_detr.class_labels_classifier(transformer_outputs.last_hidden_state)
        pred_boxes = self.raw_detr.bbox_predictor(transformer_outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            outputs_loss = {"pred_logits": logits, "pred_boxes": pred_boxes}
            loss_dict = self.raw_detr.criterion(outputs_loss, labels)
            weight_dict = self.raw_detr.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return DetrObjectDetectionOutput(
            loss=loss,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=None,
            last_hidden_state=transformer_outputs.last_hidden_state,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )