import pdb
import time
import itertools
import os
import sys
import pagexml
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
import numpy as np
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



def get_transcript(image_id,data,retinanet,score_threshold,nms_threshold,dataset_val,alphabet):
    image_name = image_id+'.jpg'
    retinanet.training=False    
    gtxml_name = os.path.join(image_name.split('/')[-1].split('.')[-2])

    
    pxml = pagexml.PageXML()
    unnormalize = UnNormalizer()
    with torch.no_grad():
        st = time.time()
        im=data['img']
        im = im.cuda().float()
        if retinanet.module.htr_gt_box:
            scores, classification, transformed_anchors,transcriptions = retinanet([im,data['annot']])
        else:
            scores, classification, transformed_anchors,transcriptions = retinanet(im)
        idxs = np.where(scores.cpu()>score_threshold)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
        #img = np.array(255 * unnormalize(im)).copy()

        img[img<0] = 0
        img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        width = img.shape[1]
        height = img.shape[0]

        conf = pagexml.ptr_double()
        pxml.newXml('retinanet_dets',image_name,width,height)
        page = pxml.selectNth("//_:Page",0)
        reg = pxml.addTextRegion(page)
        pxml.setCoordsBBox(reg,0, 0, width, height, conf )
        line = pxml.addTextLine(reg)
        pxml.setCoordsBBox(line,0, 0, width, height, conf )
        words = []
        transcriptions= np.argmax(transcriptions.cpu(),axis=-1)
        for j in range(idxs[0].shape[0]):

            # Initialize object for setting confidence values
            box = {}
            bbox = transformed_anchors[idxs[0][j], :]
            if idxs[0][j]>=transcriptions.shape[0]: continue
            transcription = transcriptions[idxs[0][j],:]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(classification[idxs[0][j]])]
            
            
            # Add a text region to the Page
            word = pxml.addWord(line,"ID"+str(j))
            
            # Set text region bounding box with a confidence
            pxml.setCoordsBBox(word,x1, y1, x2-x1, y2-y1, conf )
            
            #pxml.setCoordsBBox( reg,x1, y1, x2-x1, y2-y1, conf )
            #transcription = transcripts[j]    
            transcription = labels_to_text(transcription,alphabet)


            # Set the text for the text region
            conf.assign(0.9)
            pxml.setTextEquiv(word, transcription, conf )

            # Add property to text region
            pxml.setProperty(word,"category" , label_name )

            # Add a second page with a text region and specific id
            #page = pxml.addPage("example_image_2.jpg", 300, 300)
            #reg = pxml.addTextRegion( page, "regA" )
            #pxml.setCoordsBBox( reg, 15, 12, 76, 128 )
            words.append(word)
        words = pxml.select('//_:Word')
        order, groups = pxml.getLeftRightTopBottomReadingOrder(words, fake_baseline=True, max_horiz_iou=1, prolong_alpha=0.0)
        line = pxml.selectNth('//_:TextLine')
        group_idx = 0
        idx_in_group=0
        transcript_pred=[]
        for n in order:
            word_idx = order.index(n)
            if idx_in_group>=groups[group_idx]:
                group_idx+=1
                idx_in_group=0
            transcript_pred.append(pxml.getTextEquiv(words[n]))
            pxml.setProperty(words[n],'word_idx',str(word_idx))
            pxml.setProperty(words[n],"line",str(group_idx))
            pxml.moveElem(words[n],line)
            idx_in_group+=1
        image_text = image_id+'.txt'
        # Write XML to file
        return " ".join(transcript_pred)
