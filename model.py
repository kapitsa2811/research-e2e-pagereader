import torch.nn as nn
import torch.autograd as ag
import torch
import os
import pdb
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes,calc_iou
from anchors import Anchors
from modules.ClassificationModel import ClassificationModel
from modules.RegressionModel import RegressionModel
from modules.PyramidFeatures import PyramidFeatures
from modules.BoxSampler import BoxSampler
from modules.RecognitionModel import RecognitionModel
from modules.NERModel import NERModel
import losses
from modules.RoIPooling import roi_pooling, adaptive_max_pool,AdaptiveMaxPool2d
import cv2
import numpy as np
#import pagexml



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BidirectionalLSTM(nn.Module):
    # Module to extract BLSTM features from convolutional feature map

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.rnn.cuda()
        self.embedding.cuda()

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers,max_boxes,score_threshold,seg_level,alphabet,train_htr,htr_gt_box,two_step=False):
        self.inplanes = 64
        self.pool_h = 4
        self.pool_w = 280
        self.forward_transcription=False
        self.max_boxes = max_boxes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.downsampling_factors = [8,16,32,64,128]
        self.epochs_only_det = 1
        self.score_threshold = score_threshold
        self.alphabet=alphabet
        self.train_htr=train_htr
        self.htr_gt_box =htr_gt_box
        self.two_step = two_step

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.anchors = Anchors(seg_level=seg_level)
        self.regressionModel = RegressionModel(num_features_in=256,num_anchors=self.anchors.num_anchors)
        self.recognitionModel = RecognitionModel(feature_size=256,pool_h=self.pool_h,alphabet_len=len(alphabet))
        self.nerModel = NERModel(feature_size=256,pool_h=self.pool_h,n_classes=num_classes)
        self.classificationModel = ClassificationModel(num_features_in=256,num_anchors=self.anchors.num_anchors, num_classes=num_classes)
        self.boxSampler = BoxSampler('train',self.score_threshold)
        #self.sorter = SortRois()
        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
        
        self.focalLoss = losses.FocalLoss()    
        self.nerLoss = losses.NERLoss()
        self.transcriptionLoss = losses.TranscriptionLoss() 
  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.recognitionModel.output.weight.data.fill_(0)

        self.recognitionModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations,criterion,iter_num = inputs
        elif self.htr_gt_box:
            img_batch, annotations = inputs
            iter_num = 100000
        else:
            img_batch = inputs
            iter_num = 1000000
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)
        if self.htr_gt_box:
            rois = annotations[0,:,:4].clone()
            selected_indices=[]
            transformed_anchors = rois.clone()
        else:
            scores,classes,transformed_anchors,selected_indices = self.boxSampler(img_batch,anchors,regression,classification,self.score_threshold)
            rois = transformed_anchors.clone()
        n_boxes_predicted=transformed_anchors.shape[0]
        
        # Only calculate the recognition branch forward if there's a limited amount of positive rois and after a predifined
        # amount of epochs only trained with detection (it works with 0 epochs only detection)    

        if (iter_num>=self.epochs_only_det and (n_boxes_predicted>1 and n_boxes_predicted<self.max_boxes)) or self.htr_gt_box:
            self.forward_transcription=True 
            pooled_features=[]
            pooled_feat_indices=[]
            transcriptions=[]
            probs_sizes=[]
            
            feature = features[0]
            downsampling_factor=self.downsampling_factors[0]
            
            # calculate pooled features and transcritpion for each box:
            for j in range(rois.shape[0]):
                pooled_feature,probs_size = roi_pooling(feature,rois[j,:4],size = (self.pool_w,self.pool_h),spatial_scale=1./downsampling_factor)
                '''roi=rois[j,:4].clone()
                roi=roi.view(1,1,4,1)
                roi=roi.repeat(1,256,1,1)'''

                transcription = self.recognitionModel(pooled_feature)
                transcriptions.append(transcription)
                #pooled_feature=torch.cat([pooled_feature,roi],dim=3)
                pooled_features.append(pooled_feature)
                probs_sizes.append(probs_size[0])

            for j in range(self.max_boxes-rois.shape[0]):
               pooled_features.append(torch.zeros([1,256, self.pool_h, self.pool_w]).cuda())
            
            pooled_features= torch.stack(pooled_features,dim=0).squeeze()

            transcription = torch.stack(transcriptions,dim=0).squeeze()    
            ner_tags = self.nerModel(pooled_features)
        else:
            self.forward_transcription=False
            transcription = torch.zeros((transformed_anchors.shape[0],1,1))
            ner_tags = torch.zeros((transformed_anchors.shape[0],1,1))
            probs_sizes=[]
            
        if self.training:
            focal_loss= self.focalLoss(classification, regression, anchors, annotations,criterion,transcription,selected_indices,probs_sizes,self.pool_w,self.htr_gt_box)
            if self.forward_transcription:
                ctc_loss=self.transcriptionLoss(classification, regression, anchors, annotations,criterion,transcription,selected_indices,probs_sizes,self.pool_w,self.htr_gt_box)
                ner_loss = self.nerLoss(classification, regression, anchors, annotations,criterion,ner_tags,selected_indices,n_boxes_predicted,self.pool_w,self.htr_gt_box)

            else:
                ctc_loss=torch.tensor(30.).cuda()
                ner_loss=torch.tensor(30.).cuda()
            return focal_loss[0],focal_loss[1],ctc_loss,ner_loss
            
        else:
            if self.htr_gt_box:
                scores = torch.ones((annotations.shape[1],1))
                classes = torch.zeros((annotations.shape[1],1))
                return [scores,classes,annotations[0,:,:4],transcription]

            #return [scores,classes,transformed_anchors,transcription]
            ner_tags=torch.argmax(ner_tags,dim=-1)[:n_boxes_predicted,...]
            ner_tags = ner_tags.view(ner_tags.numel())
            return [scores,ner_tags,transformed_anchors,transcription]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
