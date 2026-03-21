import torch
from typing import List
import numpy as np

def encode(matched: torch.Tensor,priors: torch.Tensor,variances: List[float]) -> torch.Tensor: 
    target_cxcy = ((matched[:,:2] + matched[:,2:]) / 2 - priors[:,:2]) / (priors[:,2:] * variances[0])
    target_wh = torch.log((matched[:,2:] - matched[:,:2]) / priors[:,2:]) / variances[1]

    return torch.cat([target_cxcy,target_wh],dim=1)

def intersect(box_a: torch.Tensor,box_b: torch.Tensor) -> torch.Tensor:
    size_a = box_a.shape[0]
    size_b = box_b.shape[0]

    max_xy = torch.min(
        box_a[:,2:].unsqueeze(1).expand(-1,size_b,-1),
        box_b[:,2:].unsqueeze(0).expand(size_a,-1,-1)
    )
    min_xy = torch.max(
        box_a[:,:2].unsqueeze(1).expand(-1,size_b,-1),
        box_b[:,:2].unsqueeze(0).expand(size_a,-1,-1)
    )
    intersection = torch.clamp(max_xy - min_xy,min=0)
    
    return intersection[:,:,0] * intersection[:,:,1]

def jaccard(box_a: torch.Tensor,box_b: torch.Tensor) -> torch.Tensor:
    intersection = intersect(box_a=box_a,box_b=box_b)

    area_a = ((box_a[:,2] - box_a[:,0]) * (box_a[:,3] - box_a[:,1])).unsqueeze(1).expand_as(intersection)
    area_b = ((box_b[:,2] - box_b[:,0]) * (box_b[:,3] - box_b[:,1])).unsqueeze(0).expand_as(intersection)

    return intersection / (area_a + area_b - intersection)

def landm_encode(matched: torch.Tensor,priors: torch.Tensor,variances: List[float]) -> torch.Tensor:
    matched_reshape = matched.view(matched.shape[0],5,2)
    priors_reshape = priors.unsqueeze(dim=1).expand(-1,5,-1)
    encode_landm = (matched_reshape[:,:,:] - priors_reshape[:,:,:2]) / (priors_reshape[:,:,2:] * variances[0])
    return encode_landm.view(matched.shape[0],-1)

def point_form(box: torch.Tensor) -> torch.Tensor:
    return torch.cat([box[:,:2] - box[:,2:] / 2,
                      box[:,:2] + box[:,2:] / 2 ],dim=-1)

def match(
        threshold: float,
        truths: torch.Tensor,
        priors: torch.Tensor,
        variances: List[float],
        labels: torch.Tensor,
        landms: torch.Tensor,
        loc_t: torch.Tensor,
        conf_t: torch.Tensor,
        landm_t: torch.Tensor,
        idx: int
) -> None:

    overlaps = jaccard(truths,point_form(box=priors))

    best_prior_overlap,best_prior_index = overlaps.max(dim=1,keepdim=False)
    valid_gt_mask = best_prior_overlap >= 0.2
    best_prior_index_filters = best_prior_index[valid_gt_mask]
    if best_prior_index_filters.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        landm_t[idx] = 0
        return

    best_truth_overlap,best_truth_index = overlaps.max(dim=0,keepdim=False)
    best_truth_overlap.index_fill_(dim=0,index=best_prior_index_filters,value=2)
    best_truth_index[best_prior_index] = torch.arange(truths.size(0), dtype=torch.long, device=truths.device)
    
    
    conf = labels[best_truth_index]
    conf[best_truth_overlap < threshold] = 0
    conf_t[idx] = conf

    matches = truths[best_truth_index]
    loc = encode(matched=matches,priors=priors,variances=variances)
    loc_t[idx] = loc

    landm_matches = landms[best_truth_index]
    landm = landm_encode(matched=landm_matches,priors=priors,variances=variances)
    landm_t[idx] = landm

def matrix_iof(box_a: np.ndarray,box_b: np.ndarray) -> np.ndarray:
    lt = np.maximum(box_a[:,None,:2],box_b[None,:,:2])
    rb = np.minimum(box_a[:,None,2:],box_b[None,:,2:])

    wh = np.maximum(rb - lt,0.0)

    area_i = np.prod(wh,axis=2)

    area_a = np.prod(box_a[:,2:] - box_a[:,:2],axis=1,keepdims=True)

    return area_i / np.maximum(area_a,1e-5)

def decode(loc: torch.Tensor,priors: torch.Tensor,variances: List[float]) -> torch.Tensor:
    boxes_center = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes_wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    boxes = torch.cat((boxes_center, boxes_wh), dim=1)
    boxes[:, :2] -= boxes[:, 2:] / 2  
    boxes[:, 2:] += boxes[:, :2]      
    return boxes

def landms_decode(pred: torch.Tensor,priors: torch.Tensor,variances: List[float]) -> torch.Tensor:
    landms = torch.cat((
        priors[:, :2] + pred[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pred[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pred[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pred[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pred[:, 8:10] * variances[0] * priors[:, 2:]
    ), dim=1)
    return landms

def nms(boxes: torch.Tensor,scores: torch.Tensor,iou_threshold: float = 0.4) -> List:
    # boxes: [N,4]
    # scores: [N]

    _,order = scores.sort(0,descending=True)

    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        iou = jaccard(box_a=boxes[i:i + 1,:],box_b=boxes[order[1:],:]).squeeze(0)

        index = torch.where(iou <= iou_threshold)[0] # attention the function's return value

        order = order[index + 1]

    return keep

