import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou
from scipy.optimize import linear_sum_assignment
# Fallback imports nếu torchmetrics không có sẵn
try:
    from torchmetrics.detection import MeanAveragePrecision
    from torchmetrics.functional import pairwise_cosine_similarity
except ImportError:
    # Fallback implementation
    def pairwise_cosine_similarity(x, y):
        """Fallback implementation của pairwise cosine similarity"""
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        return torch.matmul(x_norm, y_norm.transpose(-2, -1))
    
    class MeanAveragePrecision:
        def __init__(self, *args, **kwargs):
            pass


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2, region_dim=256, text_dim=512):  # Reduced class cost
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.region_dim = region_dim
        self.text_dim = text_dim
        
        # Projection layer to match dimensions
        self.region_proj = nn.Linear(region_dim, 256) if region_dim != 256 else nn.Identity()
        self.text_proj = nn.Linear(text_dim, 256) if text_dim != 256 else nn.Identity()
    
    def forward(self, outputs, targets, text_embeddings):
        B, num_queries = outputs['bbox_pred'].shape[:2]
        
        region_embeddings = outputs['region_features'].flatten(0, 1)
        
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)
        
        out_bbox = outputs['bbox_pred'].flatten(0, 1)
        
        all_tgt_boxes = []
        all_tgt_labels = []
        
        for b in range(B):
            if len(targets['boxes'][b]) > 0:
                all_tgt_boxes.append(targets['boxes'][b])
                n_targets = len(targets['boxes'][b])
                all_tgt_labels.append(text_embeddings[b:b+1].expand(n_targets, -1))
        
        if len(all_tgt_boxes) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(B)]
        
        tgt_bbox = torch.cat(all_tgt_boxes)
        tgt_text_emb = torch.cat(all_tgt_labels)
        
        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        tgt_text_emb = tgt_text_emb.to(device)
        
        # Project to same dimension
        region_emb_proj = self.region_proj(region_embeddings)
        tgt_text_proj = self.text_proj(tgt_text_emb)
        
        region_emb_norm = F.normalize(region_emb_proj, dim=-1)
        tgt_text_norm = F.normalize(tgt_text_proj, dim=-1)
        
        # Sử dụng torchmetrics cho cosine similarity - tối ưu hơn
        # pairwise_cosine_similarity expects (N, D) and (M, D) -> (N, M)
        cost_class = -pairwise_cosine_similarity(region_emb_norm, tgt_text_norm)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        
        C = (self.cost_class * cost_class + 
             self.cost_bbox * cost_bbox + 
             self.cost_giou * cost_giou)
        C = C.view(B, num_queries, -1).cpu()
        
        sizes = [len(v) for v in targets['boxes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), 
                torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DetectionLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        region_dim = config.get('region_dim', 256) if config else 256
        text_dim = config.get('text_dim', 512) if config else 512
        
        self.matcher = HungarianMatcher(
            cost_class=config.get('cost_class', 1) if config else 1,
            cost_bbox=config.get('cost_bbox', 5) if config else 5,
            cost_giou=config.get('cost_giou', 2) if config else 2,
            region_dim=region_dim,
            text_dim=text_dim
        )
        
        self.lambda_cls = config.get('lambda_cls', 2.0) if config else 2.0
        self.lambda_bbox = config.get('lambda_bbox', 5.0) if config else 5.0
        self.lambda_giou = config.get('lambda_giou', 2.0) if config else 2.0
        self.lambda_sim = config.get('lambda_sim', 1.0) if config else 1.0
        self.temperature = config.get('temperature', 0.07) if config else 0.07
        
    def forward(self, outputs, targets, text_embeddings):
        indices = self.matcher.forward(outputs, targets, text_embeddings)
        
        losses = {}
        
        if hasattr(outputs, 'cls_pred') and outputs['cls_pred'] is not None:
            losses['loss_cls'] = self.lambda_cls * self.get_loss_cls(outputs, targets, indices)
        
        bbox_losses = self.get_loss_bbox_giou(outputs, targets, indices)
        losses['loss_bbox'] = self.lambda_bbox * bbox_losses['l1']
        losses['loss_giou'] = self.lambda_giou * bbox_losses['giou'] 
        
        losses['loss_sim'] = self.lambda_sim * self.get_contrastive_loss(
            outputs, targets, indices, text_embeddings)
        
        return losses
    
    def get_loss_cls(self, outputs, targets, indices):
        src_logits = outputs['cls_pred']
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.zeros_like(src_logits[:, :, 0])
        if len(idx[0]) > 0:
            target_classes[idx] = 1.0
        
        return F.binary_cross_entropy_with_logits(src_logits[:, :, 0], target_classes)
    
    def get_loss_bbox_giou(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        if len(idx[0]) == 0:
            return {
                'l1': torch.tensor(0.0, device=outputs['bbox_pred'].device, requires_grad=True),
                'giou': torch.tensor(0.0, device=outputs['bbox_pred'].device, requires_grad=True)
            }
        
        src_boxes = outputs['bbox_pred'][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets['boxes'], indices)])
        
        loss_l1 = F.l1_loss(src_boxes, target_boxes)
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes)).mean()
        
        return {'l1': loss_l1, 'giou': loss_giou}
    
    def get_contrastive_loss(self, outputs, targets, indices, text_embeddings):
        region_features = outputs['region_features']
        
        all_pos_regions = []
        all_pos_texts = []
        
        for b in range(len(indices)):
            pred_idx, tgt_idx = indices[b]
            
            if len(pred_idx) == 0:
                continue
                
            matched_regions = region_features[b][pred_idx]
            
            if text_embeddings.dim() == 3:
                text_emb = text_embeddings[b].mean(dim=0, keepdim=True)
            else:
                text_emb = text_embeddings[b:b+1]
            
            batch_text = text_emb.expand(len(matched_regions), -1)
            
            all_pos_regions.append(matched_regions)
            all_pos_texts.append(batch_text)
        
        if len(all_pos_regions) == 0:
            return torch.tensor(0.0, device=region_features.device, requires_grad=True)
        
        pos_regions = torch.cat(all_pos_regions, dim=0)
        pos_texts = torch.cat(all_pos_texts, dim=0)
        
        # Simplified negative sampling - only use a subset
        all_neg_regions = []
        for b in range(len(indices)):
            pred_idx, _ = indices[b]
            
            all_queries = torch.arange(region_features.shape[1], device=region_features.device)
            pred_idx = pred_idx.to(region_features.device)
            neg_mask = ~torch.isin(all_queries, pred_idx)
            neg_regions = region_features[b][neg_mask]
            
            # Sample only a subset of negatives to reduce computation
            if len(neg_regions) > 10:
                neg_indices = torch.randperm(len(neg_regions), device=neg_regions.device)[:10]
                neg_regions = neg_regions[neg_indices]
            
            all_neg_regions.append(neg_regions)
        
        if len(all_neg_regions) > 0:
            neg_regions = torch.cat(all_neg_regions, dim=0)
            all_regions = torch.cat([pos_regions, neg_regions], dim=0)
        else:
            all_regions = pos_regions
        
        # Ensure tensors are on the same device as the model
        device = next(self.matcher.parameters()).device
        all_regions = all_regions.to(device)
        pos_texts = pos_texts.to(device)
        
        # Project to same dimension
        regions_proj = self.matcher.region_proj(all_regions)
        texts_proj = self.matcher.text_proj(pos_texts)
        
        regions_norm = F.normalize(regions_proj, dim=-1)
        texts_norm = F.normalize(texts_proj, dim=-1)
        
        # Sử dụng torchmetrics cho cosine similarity - tối ưu hơn
        # pairwise_cosine_similarity expects (N, D) and (M, D) -> (N, M)
        sim_matrix = pairwise_cosine_similarity(texts_norm, regions_norm) / self.temperature
        labels = torch.arange(len(pos_texts), device=sim_matrix.device)
        
        return F.cross_entropy(sim_matrix, labels)
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx