# src/eval/coco_evaluator.py
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class GenericCOCOEvaluator:
    def __init__(self, gt_json_path):
        """
        Initialize the generic evaluator with ground-truth annotations.
        :param gt_json_path: Path to the standard COCO ground-truth JSON file.
        """
        self.coco_gt = COCO(gt_json_path)

    def evaluate_predictions(self, standardized_preds, output_dir):
        """
        Evaluate standardized model predictions and extract scaled mAP@0.5 metrics.
        :param standardized_preds: A list of dicts matching the official COCO results format:
                                   [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}]
        :param output_dir: Directory where the 'size_results.json' file will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save predictions to a temporary JSON file required by pycocotools API
        temp_json_path = os.path.join(output_dir, "temp_predictions.json")
        with open(temp_json_path, 'w') as f:
            json.dump(standardized_preds, f)

        # 2. Load results via pycocotools
        coco_dt = self.coco_gt.loadRes(temp_json_path)
        
        # 3. Initialize COCOeval object for bounding box detection
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        
        # 4. Execute full COCO evaluation suite
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()  # Prints the standard 12-row COCO summary to terminal

        # 5. Extract specific diagnostic metrics at mAP@0.5 (IoU threshold index 1)
        # Precision matrix dimensions: [Tx, Rx, Kx, Ax, Mx]
        # IoU threshold index [1] corresponds to IoU=0.50 (mAP@0.5)
        # Area index (Ax): 0=all, 1=small (<32^2 px), 2=medium, 3=large
        precision_matrix = coco_eval.eval['precision']
        
        extracted_metrics = {
            "mAP_50_all":    float(precision_matrix[1, :, :, 0, :].mean()),
            "mAP_50_small":  float(precision_matrix[1, :, :, 1, :].mean()),
            "mAP_50_medium": float(precision_matrix[1, :, :, 2, :].mean()),
            "mAP_50_large":  float(precision_matrix[1, :, :, 3, :].mean()),
        }

        # 6. Export results to a structured JSON file for future visualization
        results_json_path = os.path.join(output_dir, "size_results.json")
        with open(results_json_path, 'w') as f:
            json.dump(extracted_metrics, f, indent=4)
            
        print(f"\n[Evaluation Complete] Size-specific metrics saved to {results_json_path}")
        
        # Clean up the temporary file
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
            
        return extracted_metrics