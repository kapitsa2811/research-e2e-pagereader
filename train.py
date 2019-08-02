import editdistance
import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval
from get_transcript import get_transcript

from warpctc_pytorch import CTCLoss
#from torch_baidu_ctc import CTCLoss
#assert torch.__version__.split('.')[1] == '4'

print(('CUDA available: {}'.format(torch.cuda.is_available())))


def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.',default = "csv")
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',default="binary_class.csv")
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=500)
    parser.add_argument('--epochs_only_det', help='Number of epochs to train detection part', type=int, default=1)
    parser.add_argument('--max_epochs_no_improvement', help='Max epochs without improvement',type=int,default=100)
    parser.add_argument('--pretrained_model', help='Path of .pt file with pretrained model',default = 'esposallescsv_retinanet_0.pt')
    parser.add_argument('--model_out', help='Path of .pt file with trained model to save',default = 'trained')

    parser.add_argument('--score_threshold', help='Score above which boxes are kept',type=float,default=0.5)
    parser.add_argument('--nms_threshold', help='Score above which boxes are kept',type=float,default=0.2)
    parser.add_argument('--max_boxes', help='Max boxes to be fed to recognition',default=95)
    parser.add_argument('--seg_level', help='[line, word], to choose anchor aspect ratio',default='word')
    parser.add_argument('--early_stop_crit', help='Early stop criterion, detection (map) or transcription (cer)',default='cer')
    parser.add_argument('--max_iters_epoch', help='Max steps per epoch (for debugging)',default=1000000)
    parser.add_argument('--train_htr',help='Train recognition or not',default='True')
    parser.add_argument('--train_det',help='Train detection or not',default='True')
    parser.add_argument('--htr_gt_box',help='Train recognition branch with box gt (for debugging)',default='False')
    
    parser = parser.parse_args(args)

    if parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train')

        dataset_name = parser.csv_train.split("/")[-2]
        
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # Files for training log

    experiment_id =str(time.time()).split('.')[0]
    valid_cer_f=open(experiment_id+'_valid_CER.txt','w')
    for arg in vars(parser):
        if getattr(parser, arg) is not None:
            valid_cer_f.write(str(arg)+' '+str(getattr(parser, arg))+'\n')
    valid_cer_f.close()


    
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1,drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')

    # Create the model
    
    train_htr = parser.train_htr=='True'
    htr_gt_box = parser.htr_gt_box=='True'
    torch.backends.cudnn.benchmark= False 
    

    alphabet=dataset_train.alphabet
    if os.path.exists(parser.pretrained_model):
            retinanet = torch.load(parser.pretrained_model)
    else:
        if parser.depth == 18:
            retinanet = model.resnet18(
                    num_classes=dataset_train.num_classes(), 
                    pretrained=True,
                    max_boxes=int(parser.max_boxes),
                    score_threshold=float(parser.score_threshold),
                    seg_level=parser.seg_level,
                    alphabet=alphabet,
                    train_htr=train_htr,
                    htr_gt_box=htr_gt_box)

        elif parser.depth == 34:

            retinanet = model.resnet34(
                    num_classes=dataset_train.num_classes(), 
                    pretrained=True,
                    max_boxes=int(parser.max_boxes),
                    score_threshold=float(parser.score_threshold),
                    seg_level=parser.seg_level,
                    alphabet=alphabet,
                    train_htr=train_htr,
                    htr_gt_box=htr_gt_box)

        elif parser.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 152:
            retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')        

    use_gpu = True
    train_htr=parser.train_htr=='True'
    train_det=parser.train_det=='True'
    retinanet.htr_gt_box=parser.htr_gt_box=='True'

    retinanet.train_htr=train_htr
    retinanet.epochs_only_det = parser.epochs_only_det

    if use_gpu:
        retinanet = retinanet.cuda()
    
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    
    retinanet.training = True
    
    

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    ctc = CTCLoss()
    retinanet.train()
    retinanet.module.freeze_bn()
    
    best_cer = 1000
    best_map = 0
    epochs_no_improvement=0
    verbose_each=1
    optimize_each =1
    print(('Num training images: {}'.format(len(dataset_train))))

    


    for epoch_num in range(parser.epochs):
        cers=[]

        retinanet.training=True

        retinanet.train()
        retinanet.module.freeze_bn()
        
        epoch_loss = []
        
        for iter_num, data in enumerate(dataloader_train):
            if iter_num>int(parser.max_iters_epoch): break
            try:
                if iter_num % optimize_each==0:
                    optimizer.zero_grad()
                (classification_loss, regression_loss,ctc_loss,ner_loss) = retinanet([data['img'].cuda().float(), data['annot'],ctc,epoch_num])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                if train_det:    

                    if train_htr:
                        loss = ctc_loss+ classification_loss+regression_loss+ner_loss

                    else:
                        loss = classification_loss+regression_loss
                        
                elif train_htr:
                    loss = ctc_loss

                else:
                    continue
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                if iter_num % verbose_each==0:
                    print(('Epoch: {} | Step: {} |Classification loss: {:1.5f} | Regression loss: {:1.5f} | CTC loss: {:1.5f} | NER loss: {:1.5f} | Running loss: {:1.5f} | Total loss: {:1.5f}\r'.format(epoch_num,iter_num, float(classification_loss), float(regression_loss),float(ctc_loss),float(ner_loss),np.mean(loss_hist),float(loss),"\r")))
                torch.cuda.empty_cache() 

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                
            except Exception as e:
                print(e)
                continue
        if parser.dataset == 'csv' and parser.csv_val is not None and train_det:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet,score_threshold=parser.score_threshold)
            mAP=float(mAP[0][0])        

        retinanet.eval()
        retinanet.training=False    
        retinanet.score_threshold = float(parser.score_threshold) 
        for idx,data in enumerate(dataloader_val):
            if idx>int(parser.max_iters_epoch): break
            print("Eval CER on validation set:",idx,"/",len(dataset_val),"\r")
            image_name = dataset_val.image_names[idx].split('/')[-1].split('.')[-2]

            #generate_pagexml(image_name,data,retinanet,parser.score_threshold,parser.nms_threshold,dataset_val)
            text_gt = dataset_val.image_names[idx].split('.')[0]+'.txt'
            f =open(text_gt,'r')
            text_gt_lines=f.readlines()[0]
            transcript_pred = get_transcript(image_name,data,retinanet,float(parser.score_threshold),float(parser.nms_threshold),dataset_val,alphabet)
            cers.append(float(editdistance.eval(transcript_pred,text_gt_lines))/len(text_gt_lines))

        t=str(time.time()).split('.')[0]

        valid_cer_f=open(experiment_id+'_valid_CER.txt','a')
        valid_cer_f.write(str(epoch_num)+" "+str(np.mean(cers))+" "+t+'\n')
        valid_cer_f.close()
        print("GT",text_gt_lines)
        print("PREDS SAMPLE:",transcript_pred)
        

        if parser.early_stop_crit=='cer':

            if float(np.mean(cers))<float(best_cer): 
                best_cer=np.mean(cers)
                epochs_no_improvement=0
                torch.save(retinanet.module, 'trained_models/'+parser.model_out+'{}_retinanet.pt'.format(parser.dataset))
            else: epochs_no_improvement+=1
        elif parser.early_stop_crit=='map':
            if mAP>best_map:
                best_map=mAP    
                epochs_no_improvement=0
                torch.save(retinanet.module, 'trained_models/'+parser.model_out+'{}_retinanet.pt'.format(parser.dataset))
        
            else: epochs_no_improvement+=1
        if train_det:
            print(epoch_num,"mAP: ",mAP," best mAP",best_map)
        if train_htr:
            print("VALID CER:",np.mean(cers),"best CER",best_cer)    
        print("Epochs no improvement:",epochs_no_improvement)
        if epochs_no_improvement>3:
            for param_group in optimizer.param_groups:
                if param_group['lr']>10e-5:
                    param_group['lr']*=0.1
        
        if epochs_no_improvement>=parser.max_epochs_no_improvement:
            print("TRAINING FINISHED AT EPOCH",epoch_num,".")
            sys.exit()
        
        scheduler.step(np.mean(epoch_loss))    
        torch.cuda.empty_cache()    
        

    retinanet.eval()

    #torch.save(retinanet, 'model_final.pt'.format(epoch_num))

if __name__ == '__main__':
 main()
