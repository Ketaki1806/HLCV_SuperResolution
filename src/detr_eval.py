import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor
from tqdm import tqdm

from src.models.detr_anyup import DetrWithAnyUpSwitch
from src.eval.coco_evaluator import GenericCOCOEvaluator
import src.models.anyup as anyup

# ==============================================================================
# 1. Loading COCO dataset and preparing batch inputs for DETR
# ==============================================================================
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def detr_collate_fn(batch):
    """
    Collate data returned by torchvision.datasets.CocoDetection into a batch 
    that fits Hugging Face DETR inputs.
    """
    # Internal batch structure: [(PIL_Image, target_annotations), ...]
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Extract image IDs and ground-truth original heights/widths (for coordinate scaling)
    # Default to None if an image has no annotations (though they are usually present in val2017)
    image_ids = [t[0]['image_id'] if len(t) > 0 else None for t in targets]
    
    # PIL size is (width, height) -> convert to (height, width) to match PyTorch conventions
    original_sizes = [(img.size[1], img.size[0]) for img in images] 
    
    # Leverage HF processor to automatically handle multi-image padding alignment and generate pixel_mask
    inputs = processor(images=images, return_tensors="pt")
    
    return {
        "pixel_values": inputs["pixel_values"],
        "pixel_mask": inputs["pixel_mask"],
        "image_id": image_ids,
        "original_size": original_sizes
    }


# ==============================================================================
# 2. Post-processing DETR outputs to standard COCO format
# ==============================================================================
def postprocess_detr_outputs(logits, pred_boxes, image_ids, original_sizes):
    """
    Convert normalized relative coordinates [cx, cy, w, h] from DETR outputs 
    into absolute pixel coordinates [x_min, y_min, width, height] required by COCO.
    """
    standardized_preds = []
    
    # Convert logits to probability distributions over classes
    probs = logits.softmax(-1)
    
    # Get maximum confidence scores and corresponding class labels (excluding the background class at the last index)
    scores, labels = probs[..., :-1].max(-1) 
    
    # Loop over the batch dimension
    for b in range(logits.shape[0]):
        img_id = image_ids[b]
        if img_id is None:
            continue
        img_id = int(img_id)
        orig_h, orig_w = original_sizes[b]
        
        # Loop over the 100 queries (prediction slots)
        for q in range(logits.shape[1]):
            score = float(scores[b, q].item())
            
            # Filter out extremely low-confidence predictions to drastically reduce temp JSON file size
            if score < 0.05:
                continue
                
            category_id = int(labels[b, q].item())
            
            # Unbind coordinates: [cx, cy, w, h]
            cx, cy, w, h = pred_boxes[b, q].unbind(-1)
            
            # Mathematical conversion: 
            # [cx, cy, w, h] relative coordinates -> [x_min, y_min, width, height] absolute bounding box
            x_min = (cx - 0.5 * w) * orig_w
            y_min = (cy - 0.5 * h) * orig_h
            width = w * orig_w
            height = h * orig_h
            
            standardized_preds.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [
                    float(x_min.item()), 
                    float(y_min.item()), 
                    float(width.item()), 
                    float(height.item())
                ],
                "score": score
            })
            
    return standardized_preds


# ==============================================================================
# 3. Main Evaluation Pipeline: Sequential evaluation of the four strategies
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths according to GPU cluster storage conventions
    SCRATCH_DIR = "/scratch/teaching/hlcv/hlcv019/data" 
    GT_JSON_PATH = os.path.join(SCRATCH_DIR, "annotations/instances_val2017.json")
    
    # Initialize the custom COCO evaluator based on official pycocotools
    evaluator = GenericCOCOEvaluator(gt_json_path=GT_JSON_PATH)
    
    # Configuration details
    MODEL_NAME_OR_PATH = "facebook/detr-resnet-50"
    detector = "detr_zero_shot"
    strategies = ["baseline", "espcn2x", "fsrcnn2x", "anyup"]
    
    # Iterate through each zero-shot evaluation strategy
    for strategy in strategies:
        model_strat = f"{detector}_{strategy}"
        print(f"\n" + "="*70)
        print(f"Launching Zero-Shot evaluation for: {model_strat}")
        print("="*70)
        
        # Output and prediction directory: runs/model_strat/
        output_dir = f"runs/{model_strat}"
        os.makedirs(output_dir, exist_ok=True)
        
        # A. Select the corresponding image folder depending on the strategy
        if strategy in ["baseline", "anyup"]:
            image_dir = os.path.join(SCRATCH_DIR, "preprocessed/low_res/scale_x2/test_val2017")
        elif strategy == "espcn2x":
            image_dir = os.path.join(SCRATCH_DIR, "preprocessed/espcn_x2/test_val2017")
        elif strategy == "fsrcnn2x":
            image_dir = os.path.join(SCRATCH_DIR, "preprocessed/fsrcnn_x2/test_val2017")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        # B. Load dataset via torchvision CocoDetection
        print(f"-> Loading image dataset source from: {image_dir}")
        dataset = CocoDetection(root=image_dir, annFile=GT_JSON_PATH)
        dataloader = DataLoader(
            dataset, 
            batch_size=4,        # Batched inference; can be increased to 8 if VRAM permits
            shuffle=False, 
            num_workers=4, 
            collate_fn=detr_collate_fn
        )
        
        # C. Load the official AnyUp model dynamically if needed
        # We replace the dummy mock layer with the real pre-trained AnyUp module
        if strategy == "anyup":
            print("-> [Loading] Fetching pre-trained AnyUp weights from torch.hub...")
            anyup_module = anyup.load_anyup(device=device)
        else:
            # For non-anyup strategies, we can pass None to save GPU memory and initial startup time
            anyup_module = None
            
        # Instantiate DETR wrapper with dynamic AnyUp support
        model = DetrWithAnyUpSwitch(
            model_name_or_path=MODEL_NAME_OR_PATH, 
            anyup_model=anyup_module
        )
        
        # D. Configure internal model switch flags
        if strategy == "anyup":
            model.use_anyup = True
            print("-> [Status] AnyUp (adaptive 2x feature upsampling) is now ACTIVATED.")
        else:
            model.use_anyup = False
            print("-> [Status] AnyUp is DEACTIVATED. Running standard forward pass.")
            
        model.to(device)
        model.eval()
        
        # E. Batch Inference
        all_standardized_preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Strategy: {strategy}"):
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                image_ids = batch["image_id"]
                original_sizes = batch["original_size"]
                
                # Perform forward pass
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                
                # Format model predictions and append to collector
                preds = postprocess_detr_outputs(
                    logits=outputs["logits"],
                    pred_boxes=outputs["pred_boxes"],
                    image_ids=image_ids,
                    original_sizes=original_sizes
                )
                all_standardized_preds.extend(preds)
                
        # F. Dump normalized predictions to runs/model_strat/predictions.json
        preds_save_path = os.path.join(output_dir, "predictions.json")
        with open(preds_save_path, 'w') as f:
            json.dump(all_standardized_preds, f)
        print(f"-> [Saved] Predictions successfully exported to: {preds_save_path}")
        
        # G. Call GenericCOCOEvaluator to print standard 12-line summary 
        # and dump size_results.json into the output folder
        evaluator.evaluate_predictions(all_standardized_preds, output_dir=output_dir)

if __name__ == "__main__":
    main()