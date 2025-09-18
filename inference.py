"""
inference with sam 
"""

from segment_anything import SamPredictor, sam_model_registry

import json 

def inference(js_file_path, model_checkpoint='<path/to/checkpoint>', model_type='',):
    res = [] 
    with open(js_file_path, 'r') as js_file: 
        box_predictions = json.load(js_file)
    sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
    predictor = SamPredictor(sam)
    for pred in box_predictions:
        predictor.set_image(pred['filename'])
        masks, seg_scores, seg_logits = predictor.predict(pred['pred_boxes'])
        res.append({'pred_masks':masks
                    , 'pred_seg_scores':seg_scores
                    , 'pred_seg_logits':seg_logits
                    , 'filename': pred['filename']
                    , 'pred_boxes': pred['pred_boxes']
                    , 'pred_box_scores': pred['scores']
                    })
    
    return res 

from torchvision.ops import nms
import torch 
def NMS(js_file_path, iou_thr): 
    """
    non maximum suppression for overlapping predicted boxes 
    """
    res = []
    with open(js_file_path, 'r') as js_file: 
        predictions = json.load(js_file)
    for pred in predictions: 
        boxes = pred['pred_boxes']
        scores = pred['scores']
        
        keep_indices = nms(boxes=torch.tensor(boxes), scores=torch.tensor(scores), iou_threshold=iou_thr)
        print(keep_indices)
        # combined = list(zip(boxes, scores))
        # combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        # sorted_boxes, sorted_scores = zip(*combined_sorted)

        # keep_boxes = sorted_boxes[keep_indices]
        # keep_scores = sorted_scores[keep_indices]
        # res.append({'filename':pred['filename'], 'pred_boxes':keep_boxes, 'scores': keep_scores}) 


