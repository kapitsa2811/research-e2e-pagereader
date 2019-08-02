import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import pagexml
import sys
import cv2
import itertools

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from utils import labels_to_text
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer



print(('CUDA available: {}'.format(torch.cuda.is_available())))





def generate_pagexml(image_id,data,retinanet,score_threshold,nms_threshold,dataset_val):
    image_name = image_id+'.jpg'
    im_file_out ='pagexmls/'+image_name
    alphabet = retinanet.alphabet
    #retinanet.score_threshold = torch.tensor(score_threshold).cuda().float()
    colors = get_n_random_colors(len(dataset_val.labels))
    gtxml_name = os.path.join(image_name.split('/')[-1].split('.')[-2])

    
    pxml = pagexml.PageXML()
    unnormalize = UnNormalizer()
    with torch.no_grad():
        st = time.time()
        im=data['img']
        
        im = im.cuda().float()
        print(retinanet.htr_gt_box)
        if retinanet.htr_gt_box:
            scores, classification, transformed_anchors,transcriptions = retinanet([im,data['annot']])
            score_threshold=0
        else:
            scores, classification, transformed_anchors,transcriptions = retinanet(im)
    
        n_boxes_predicted = transformed_anchors.shape[0]
        print(n_boxes_predicted,"BOXES PREDICTED")

        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

        img[img<0] = 0
        img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        width = img.shape[1]
        height = img.shape[0]
        cv2.imwrite(im_file_out,img)

        conf = pagexml.ptr_double()
        pxml.newXml('retinanet_dets',image_name,width,height)
        page = pxml.selectNth("//_:Page",0)
        reg = pxml.addTextRegion(page)
        pxml.setCoordsBBox(reg,0, 0, width, height, conf )
        line = pxml.addTextLine(reg)
        pxml.setCoordsBBox(line,0, 0, width, height, conf )
        words = []
        for k in range(len(dataset_val.labels)):
            cv2.putText(img,dataset_val.labels[k],(25,25+k*15), cv2.FONT_HERSHEY_PLAIN, 1, colors[k], 2)
        transcriptions= np.argmax(transcriptions.cpu(),axis=-1)
        for box_id in range(n_boxes_predicted):

            # Initialize object for setting confidence values
            box = {}
            bbox = transformed_anchors[box_id, :]
            transcription = transcriptions[box_id,:]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(classification[box_id])]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color=colors[int(classification[box_id])], thickness=2)
            
            # Add a text region to the Page
            word = pxml.addWord(line,"ID"+str(box_id))
            
            # Set text region bounding box with a confidence
            pxml.setCoordsBBox(word,x1, y1, x2-x1, y2-y1, conf )
            
            #pxml.setCoordsBBox( reg,x1, y1, x2-x1, y2-y1, conf )
            #transcription = transcripts[j]    
            transcription = labels_to_text(transcription,alphabet)
            draw_caption(img, (x1, y1, x2, y2), transcription)


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
        for n in order:
            word_idx = order.index(n)
            if idx_in_group>=groups[group_idx]:
                group_idx+=1
                idx_in_group=0

            pxml.setProperty(words[n],'word_idx',str(word_idx))
            pxml.setProperty(words[n],"line",str(group_idx))
            pxml.moveElem(words[n],line)
            idx_in_group+=1

        # Write XML to file
        pxml.write('pagexmls/'+gtxml_name+".xml")
        cv2.imwrite(os.path.join('pred_sample_ims',str(image_id)+'.jpg'), img)


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def get_n_random_colors(n):
    colors = []
    #for i in range(n):
        #color = (int(255*np.random.random()),int(255*np.random.random()),int(255*np.random.random()))
        #colors.append(color)
    colors=[(255,255,255),(255,0,0),(0,255,0),(255,0,255),(0,255,255),(122,122,0)]
    return colors
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.',default="csv")
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',default="binary_class.csv")
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--csv_box_annot', help='Path to file containing predicted box annotations ')
    parser.add_argument('--score_threshold', help='Score above which boxes are kept',default=0.35)
    parser.add_argument('--nms_threshold', help='Score above which boxes are kept',default=0.2)
    parser.add_argument('--two_step', help='Detect and transcribe in separate steps',default=False)

    parser.add_argument('--htr_gt_box',help='Train recognition branch with box gt (for debugging)',default=False)
    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
    if parser.csv_box_annot is not None:
        box_annot_data = CSVDataset(train_file=parser.csv_box_annot, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        box_annot_data = None

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val,shuffle=False)
    
    if box_annot_data is not None:
        sampler_val = AspectRatioBasedSampler(box_annot_data, batch_size=1, drop_last=False)
        dataloader_box_annot = DataLoader(box_annot_data, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    else:
        dataloader_box_annot = dataloader_val

    torch.backends.cudnn.benchmark= False 
    retinanet = torch.load(parser.model)
    score_threshold = float(parser.score_threshold)
    nms_threshold = float(parser.score_threshold)
    use_gpu = True
    htr_gt_box = parser.htr_gt_box=='True'    
    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()
    retinanet.two_step = parser.two_step==True
    retinanet.epochs_only_det = 0
    retinanet.htr_gt_box = htr_gt_box
    if not os.path.exists('pagexmls'):
        os.mkdir('pagexmls')
    if not os.path.exists('pred_sample_ims'):
        os.mkdir('pred_sample_ims')

    for idx, data in enumerate(dataloader_box_annot):
        if box_annot_data:
            image_name = box_annot_data.image_names[idx].split('/')[-1].split('.')[-2]
        else:    
            image_name = dataset_val.image_names[idx].split('/')[-1].split('.')[-2]
        # Create a new page xml
        generate_pagexml(image_name,data,retinanet,score_threshold,nms_threshold,dataset_val)
        print("Get more preds?")
        '''continue_eval =raw_input()
        if continue_eval!='n' and continue_eval!='N': continue
        else: sys.exit()'''

if __name__ == '__main__':
 main()
