# src/eval/plotter.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_four_strategy_bars(baseline_json, espcn_json, fsrcnn_json, anyup_json, output_dir="runs"):
    """
    Plots a side-by-side bar chart with 4 bars per category to compare all SR strategies.
    """
    # 1. Load 4 data sources
    with open(baseline_json, 'r') as f: base = json.load(f)
    with open(espcn_json, 'r') as f: espcn = json.load(f)
    with open(fsrcnn_json, 'r') as f: fsrcnn = json.load(f)
    with open(anyup_json, 'r') as f: anyup = json.load(f)
        
    categories = ['All', 'Small (<32²)', 'Medium (32²-96²)', 'Large (>96²)']
    
    # Extract and convert to percentages
    s_base = [base['mAP_50_all']*100, base['mAP_50_small']*100, base['mAP_50_medium']*100, base['mAP_50_large']*100]
    s_espcn = [espcn['mAP_50_all']*100, espcn['mAP_50_small']*100, espcn['mAP_50_medium']*100, espcn['mAP_50_large']*100]
    s_fsrcnn = [fsrcnn['mAP_50_all']*100, fsrcnn['mAP_50_small']*100, fsrcnn['mAP_50_medium']*100, fsrcnn['mAP_50_large']*100]
    s_anyup = [anyup['mAP_50_all']*100, anyup['mAP_50_small']*100, anyup['mAP_50_medium']*100, anyup['mAP_50_large']*100]
    
    # 2. Setup plotting dimensions for 4 bars
    x = np.arange(len(categories))
    width = 0.2  # Thinner bars to fit all 4 comfortably
    
    fig, ax = plt.subplots(figsize=(10, 6)) # Wider canvas for 4 bars
    
    # Position each bar with a specific offset around the center 'x'
    rects1 = ax.bar(x - 1.5*width, s_base, width, label='Baseline', color='#7F8C8D')
    rects2 = ax.bar(x - 0.5*width, s_espcn, width, label='with ESPCN 2x', color='#A9CCE3')
    rects3 = ax.bar(x + 0.5*width, s_fsrcnn, width, label='with FSRCNN 2x', color='#5499C7')
    rects4 = ax.bar(x + 1.5*width, s_anyup, width, label='with AnyUp (Ours)', color='#E74C3C') # Vibrant Red/Orange for focus
    
    # 3. Academic styling
    ax.set_ylabel('mAP @ 0.5 (%)', fontsize=12)
    ax.set_title('Performance Comparison Across Different SR and Scale Strategies', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # (Optional) You can call the autolabel function here if the text font is small enough to not overlap.
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "four_strategy_comparison.png"), dpi=300)
    plt.close()

def plot_cross_strategy_model_bars(runs_dir="runs", output_dir="runs"):
    """
    Plots a side-by-side bar chart focusing STRICTLY on Small Objects (<32² px).
    The X-axis represents the 4 strategies, comparing how different models 
    perform under the exact same resolution/scale configuration.
    """
    detectors = ["yolov8n", "faster_rcnn", "detr_zero_shot", "detr_finetuned"]
    strategies = ["baseline", "espcn2x", "fsrcnn2x", "anyup"]
    
    # Initialize storage for small object scores: {detector_name: [score_strat1, score_strat2, ...]}
    detector_scores = {det: [] for det in detectors}
    
    # 1. Scrape data from directory structure sorted by strategy then detector
    for strat in strategies:
        for det in detectors:
            folder_name = f"{det}_{strat}"
            json_path = os.path.join(runs_dir, folder_name, "size_results.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                # Extract small object mAP and convert to percentage
                detector_scores[det].append(data["mAP_50_small"] * 100)
            else:
                # Placeholder for missing or unfinished experiment runs
                detector_scores[det].append(0.0)
                
    # 2. Setup plotting dimensions (X-axis represents the 4 STRATEGIES now)
    x = np.arange(len(strategies))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(11, 6)) # Slightly wider to prevent label overlaps
    
    # Distinct academic color palette to differentiate the 4 model families
    rects1 = ax.bar(x - 1.5*width, detector_scores["yolov8n"], width, label='YOLOv8n', color='#34495E')
    rects2 = ax.bar(x - 0.5*width, detector_scores["faster_rcnn"], width, label='Faster R-CNN', color='#3498DB')
    rects3 = ax.bar(x + 0.5*width, detector_scores["detr_zero_shot"], width, label='DETR (Zero-Shot)', color='#9B59B6')
    rects4 = ax.bar(x + 1.5*width, detector_scores["detr_finetuned"], width, label='DETR (Fine-Tuned)', color='#8E44AD')
    
    # 3. Academic styling and labels
    ax.set_ylabel('Small Object mAP @ 0.5 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Model Diagnostics: Small Object Sensitivity Under Identical Strategies', fontsize=13, fontweight='bold')
    
    # X-axis ticks represent the 4 core methodologies being compared
    display_strategy_labels = ['Baseline (Raw LR)', 'with ESPCN 2x', 'with FSRCNN 2x', 'with AnyUp (Ours)']
    ax.set_xticks(x)
    ax.set_xticklabels(display_strategy_labels, fontsize=11)
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    save_path = os.path.join(output_dir, "cross_strategy_model_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Success] Cross-strategy model comparison chart saved to {save_path}")