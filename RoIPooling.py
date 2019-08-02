import torch
import torch.nn as nn
import torch.autograd as ag
import math
import pdb

from torch.autograd.function import Function
from torch._thnn import type2backend

class AdaptiveMaxPool2d(Function):
    def __init__(self, out_w, out_h):
        super(AdaptiveMaxPool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input):
    
        output = input.new()
        indices = input.new().long()
        self.save_for_backward(input)
        self.indices = indices
        self._backend = type2backend[input.type()]
        self._backend.SpatialAdaptiveMaxPooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_w, self.out_h)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        grad_input = grad_output.new()
        self._backend.SpatialAdaptiveMaxPooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices)
        return grad_input, None

def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0],size[1])(input)

def roi_pooling(input, rois, size=(100,30), spatial_scale=1.0):
    rois = rois.view(1,rois.numel())
    assert(rois.dim() == 2)
    assert(rois.size(1) == 4)
    output = []
    probs_sizes = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    rois[:,0:].mul_(spatial_scale)
    #rois = rois.long()
    #rois = torch.round(rois).long()
    #for i in range(int(num_rois/20000)):
    for i in range(int(num_rois)):
        roi = rois[i]
        roi[0]=torch.floor(roi[0])
        roi[1]=torch.floor(roi[1])

        roi[2]=torch.ceil(roi[2])
        roi[3]=torch.ceil(roi[3])
        roi = roi.long()
        im_idx =0#roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[1]:(roi[3]+1), roi[0]:(roi[2]+1)]

        if im.shape[-1]<1 or im.shape[-2]<1:
            print(roi)
            continue  
        real_width = int((roi[2]-roi[0]).cpu())
        real_height = max(int((roi[3]-roi[1]).cpu()),1)
    
        scale_h = float(size[1])/real_height
        no_padded_w = int(math.ceil(real_width*scale_h))
        # Add padding or cut roi
        if no_padded_w<1: 
            print(roi)
            continue
        if no_padded_w<size[0]:
            pooled_feat = adaptive_max_pool(im,(no_padded_w,size[1]))
            pooled_feat = torch.cat([pooled_feat,torch.zeros(pooled_feat.shape[0],pooled_feat.shape[1],pooled_feat.shape[2],size[0]-no_padded_w).cuda()],dim=-1)
            probs_sizes.append(no_padded_w)
        else:
            pooled_feat = adaptive_max_pool(im,size)
            probs_sizes.append(size[0])
        
        output.append(pooled_feat)
    if len(output)<1: 
        return torch.zeros([1,1,1,1]),torch.Tensor([0])
    return torch.cat(output, 0),torch.Tensor(probs_sizes).int()

if __name__ == '__main__':
    input = ag.Variable(torch.rand(1,1,10,10), requires_grad=True)
    rois = ag.Variable(torch.LongTensor([[0,1,2,7,8],[0,3,3,8,8]]),requires_grad=False)
    #rois = ag.Variable(torch.LongTensor([[0,3,3,8,8]]),requires_grad=False)

    out = adaptive_max_pool(input,(3,3))
    out.backward(out.data.clone().uniform_())

    out = roi_pooling(input, rois, size=(3,3))
    out.backward(out.data.clone().uniform_())


