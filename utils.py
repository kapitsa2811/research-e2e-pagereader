import cv2
import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import sys
import itertools

def labels_to_text(labels,alphabet):
    ret = []
    labels = list(labels)
    labels = [k for k, g in itertools.groupby(labels)]
    for c in labels:
        if c ==0:# len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c-1])
    ret = "".join(ret)
    
    return ret
def get_transcript_txt(csv_file):
    csv_file = open(csv_file,'r')
    reader = csv.reader(csv_file,delimiter=',')
    gt_files = {}
    for annot in reader:
            gt_txt_file = annot[0].split('.')[0]+'.txt'
            if gt_files.get(gt_txt_file) is not None:
                    gt_files[gt_txt_file].append(annot[-1])

            else: gt_files[gt_txt_file]=[annot[-1]]
	
    for k,v in gt_files.iteritems():
            f=open(k,'w')
            f.write(" ".join(v))
            f.close()

def choose_optimal_anchor_ratios(boxes_gt_file):

    f = open(boxes_gt_file,'r')
    flines = f.readlines()
    all_wh=np.zeros((len(flines),2))

    for line_idx in range(len(flines)):
        line = flines[line_idx]
        vals = line.split(',')
        x0 =int(vals[1])
        x1 =int(vals[3])
        y0 =int(vals[2])
        y1 =int(vals[4])

        all_wh[line_idx,0]=x1-x0
        all_wh[line_idx,1]=y1-y0

    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(all_wh)
    ratios=kmeans.cluster_centers_[:,1]/kmeans.cluster_centers_[:,0]
    print ("Optimal",n_centers,"ratios",ratios)


def view_feature(pooled_feature,probs_sizes,feat_id):
	max_features_vis=min(pooled_feature.shape[1],10)
	for i in range(max_features_vis):#range(pooled_feature.shape[1]):
		img = pooled_feature[0,i].detach().cpu()
		img = np.array(255*img)
		img[img<0] = 0
		img[img>255] = 255
		#img = img.transpose(1,2,0)
		#img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
		img = img[:,:probs_sizes]

		if not os.path.exists('visualized_feats'): 
			os.mkdir('visualized_feats')
		cv2.imwrite('visualized_feats/feat_id'+str(feat_id)+'_'+str(i)+'.jpg',img)	

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes
