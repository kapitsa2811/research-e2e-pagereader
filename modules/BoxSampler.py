import torch.nn as nn
import torch
from utils import BBoxTransform,ClipBoxes
from nms.nms_pytorch import nms as nms_op

def nms(dets, thresh):
    #dets = dets.cpu().detach()
    #dets = np.array(dets)
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    #keep = torch.tensor(py_cpu_nms(dets,thresh))
    scores = dets[:,4].clone()
    detections = dets[:,:4].clone()
    thresh=0.1
    keep,count = nms_op(detections,scores,thresh)
    return keep.view(keep.numel())
    #return nms_op(dets,scores, thresh)

class BoxSampler(nn.Module):
    # Module to calculate the box coordinates given the classification scores, offset regression,
    # score and nms thresholds

    def __init__(self,training,score_threshold):
        super(BoxSampler, self).__init__()
        self.training = training
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.score_threshold = score_threshold

    def forward(self,img_batch,anchors,regression,classification,score_threshold):
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores>score_threshold)[0, :, 0]
        #scores_over_thresh = (scores>score_threshold)[0, :, 0]
        
        scores_over_thresh_idx=scores_over_thresh.nonzero()
        if scores_over_thresh.sum() > 0:
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            
            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.2)
            transformed_anchors  = transformed_anchors[0,anchors_nms_idx,:]
            
            classification = classification[:, scores_over_thresh, :]
            nms_scores, classes = classification[0, anchors_nms_idx, :].max(dim=1)

            selected_indices = scores_over_thresh_idx[anchors_nms_idx]
            selected_indices = selected_indices.view(selected_indices.numel())
            return nms_scores,classes,transformed_anchors,selected_indices
        else:
            return torch.zeros(1, 1),torch.zeros(1, 1),torch.zeros(1, 4),torch.zeros(1)

