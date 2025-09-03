import torch
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision

class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.map_metric = MeanAveragePrecision()
    
    def __call__(self, outputs, targets):
        metrics = {}
        
        # Convert predictions to detection format
        pred_boxes, pred_scores, pred_labels = self.convert_predictions(outputs)
        target_boxes, target_labels = self.convert_targets(targets)
        
        # Calculate mAP
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            preds = [{"boxes": pb, "scores": ps, "labels": pl} 
                    for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels)]
            targets_map = [{"boxes": tb, "labels": tl} 
                          for tb, tl in zip(target_boxes, target_labels)]
            
            try:
                map_result = self.map_metric(preds, targets_map)
                metrics['mAP'] = map_result['map'].item()
                metrics['mAP_50'] = map_result['map_50'].item()
            except:
                metrics['mAP'] = 0.0
                metrics['mAP_50'] = 0.0
        else:
            metrics['mAP'] = 0.0
            metrics['mAP_50'] = 0.0
        
        # Classification accuracy
        cls_pred = outputs['cls_pred'].sigmoid()
        similarity = outputs['similarity'].sigmoid()
        
        metrics['cls_acc'] = (cls_pred > 0.5).float().mean().item()
        metrics['sim_acc'] = (similarity > 0.5).float().mean().item()
        
        return metrics
    
    def convert_predictions(self, outputs):
        cls_pred = outputs['cls_pred'].sigmoid()
        bbox_pred = outputs['bbox_pred']
        similarity = outputs['similarity'].sigmoid()
        
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        
        for b in range(cls_pred.shape[0]):
            # Filter confident predictions
            scores = cls_pred[b, :, 0] * similarity[b]
            keep = scores > 0.3
            
            boxes = bbox_pred[b][keep]
            scores = scores[keep]
            labels = torch.ones_like(scores, dtype=torch.long)
            
            pred_boxes.append(boxes)
            pred_scores.append(scores)
            pred_labels.append(labels)
        
        return pred_boxes, pred_scores, pred_labels
    
    def convert_targets(self, targets):
        target_boxes = []
        target_labels = []
        
        for boxes in targets['boxes']:
            if len(boxes) > 0:
                labels = torch.ones(len(boxes), dtype=torch.long)
            else:
                labels = torch.empty(0, dtype=torch.long)
            
            target_boxes.append(boxes)
            target_labels.append(labels)
        
        return target_boxes, target_labels