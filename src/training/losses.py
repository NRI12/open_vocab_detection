
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    def __call__(self, outputs, targets):
        B, num_queries = outputs['cls_pred'].shape[:2]
        
        out_prob = outputs['cls_pred'].flatten(0, 1).sigmoid()
        out_bbox = outputs['bbox_pred'].flatten(0, 1)
        
        tgt_bbox = torch.cat(targets['boxes'])
        
        cost_class = -out_prob[:, 0]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class[:, None] + self.cost_giou * cost_giou
        C = C.view(B, num_queries, -1).cpu()
        
        sizes = [len(v) for v in targets['boxes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class DetectionLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.matcher = HungarianMatcher()
        
    def forward(self, outputs, targets, texts):
        indices = self.matcher(outputs, targets)
        
        losses = {}
        losses['loss_ce'] = self.get_loss_ce(outputs, targets, indices)
        losses['loss_bbox'] = self.get_loss_bbox(outputs, targets, indices)
        losses['loss_sim'] = self.get_loss_similarity(outputs, targets, indices)
        
        return losses
    def get_loss_ce(self, outputs, targets, indices):
            src_logits = outputs['cls_pred']
            idx = self._get_src_permutation_idx(indices)
            target_classes = torch.zeros_like(src_logits[:, :, 0])
            target_classes[idx] = 1.0
            
            return F.binary_cross_entropy_with_logits(src_logits[:, :, 0], target_classes)
        
    def get_loss_bbox(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bbox_pred'][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets['boxes'], indices)])
        
        return F.smooth_l1_loss(src_boxes, target_boxes)
    
    def get_loss_similarity(self, outputs, targets, indices):
        similarity = outputs['similarity']
        idx = self._get_src_permutation_idx(indices)
        target_sim = torch.zeros_like(similarity)
        target_sim[idx] = 1.0
        
        return F.binary_cross_entropy_with_logits(similarity, target_sim)
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx