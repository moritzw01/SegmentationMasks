"""
inference with sam 
"""

from segment_anything import SamPredictor, sam_model_registry

import json 

def inference(js_file, model_checkpoint='<path/to/checkpoint>', model_type='',):
    res = [] 
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
                    , 'pred_box_scores': pred['pred_boxes']
                    })


