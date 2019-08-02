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
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))



def labels_to_text(labels,alphabet):
    ret = []
    for c in labels:
	if c ==0 or c==len(alphabet):# len(alphabet):  # CTC Blank
	    ret.append("")
	else:
	    ret.append(alphabet[c])
    return "".join(ret)
    out = np.reshape(out,(1,w,len(self.alphabet)+1))
    ret=[]	
    for j in range(out.shape[0]):
	out_best = list(np.argmax(out[j, 2:], 1))
	out_best = [k for k, g in itertools.groupby(out_best)]
	outstr = labels_to_text(out_best)
	ret.append(outstr)
	
    return ret


def generate_pagexml(image_id,data,retinanet,score_threshold,dataset_val,nms_threshold):
	image_name = image_id+'.jpg'
	file ='pagexmls/'+image_name
	alphabet = " abcdefghijklmnopqrstuvwxy z"
	
	colors = get_n_random_colors(len(dataset_val.labels))
	gtxml_name = os.path.join(image_name.split('/')[-1].split('.')[-2])

	
	pxml = pagexml.PageXML()
	unnormalize = UnNormalizer()
	with torch.no_grad():
		st = time.time()
		im=data['img']
		
		im = im.cuda().float()
		scores, classification, transformed_anchors = retinanet([im,nms_threshold])
		print('Elapsed time: {}'.format(time.time()-st))
		
		idxs = np.where(scores>score_threshold)
		img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
		#img = np.array(255 * unnormalize(im)).copy()

		img[img<0] = 0
		img[img>255] = 255

		img = np.transpose(img, (1, 2, 0))

		img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
		width = img.shape[1]
		height = img.shape[0]
		cv2.imwrite(file,img)

		conf = pagexml.ptr_double()
		pxml.newXml('retinanet_dets',image_name,width,height)
		page = pxml.selectNth("//_:Page",0)
		reg = pxml.addTextRegion(page)
		pxml.setCoordsBBox(reg,0, 0, width, height)
		line = pxml.addTextLine(reg)
		pxml.setCoordsBBox(line,0, 0, width, height)
		words = []
		for k in range(len(dataset_val.labels)):
			cv2.putText(img,dataset_val.labels[k],(25,25+k*15), cv2.FONT_HERSHEY_PLAIN, 1, colors[k], 2)
		for j in range(idxs[0].shape[0]):

			# Initialize object for setting confidence values
			box = {}
			bbox = transformed_anchors[idxs[0][j], :]
			x1 = int(bbox[0])
			y1 = int(bbox[1])
			x2 = int(bbox[2])
			y2 = int(bbox[3])
			label_name = dataset_val.labels[int(classification[idxs[0][j]])]

			cv2.rectangle(img, (x1, y1), (x2, y2), color=colors[int(classification[idxs[0][j]])], thickness=2)
			
			# Add a text region to the Page
			word = pxml.addWord(line,"ID"+str(j))
			
			# Set text region bounding box with a confidence
			pxml.setCoordsBBox(word,x1, y1, x2-x1, y2-y1)
			
			#pxml.setCoordsBBox( reg,x1, y1, x2-x1, y2-y1, conf )
			
			transcripts=[]
			confs=[]
			seq_len = int(bbox[4])
			for k in range(seq_len+1):
				transcripts.append(np.argmax(bbox[(5+k*27):((5+(k+1)*27))]))
			transcripts=np.array(transcripts)
			transcript=labels_to_text(transcripts,alphabet)
			draw_caption(img, (x1, y1, x2, y2), "".join([alphabet[transcripts[k]] for k in range(len(transcripts))]))


			# Set the text for the text region
			conf.assign(1)
			pxml.setTextEquiv(word, "".join([alphabet[transcripts[k]] for k in range(len(transcripts))]))

			# Add property to text region
			pxml.setProperty(word,"category" , label_name )

			words.append(word)
		words = pxml.select('//_:Word')
		order, groups = pxml.getLeftRightTopBottomReadingOrder(words, fake_baseline=True, max_horiz_iou=1, prolong_alpha=0.0)
		line = pxml.selectNth('//_:TextLine',0)
		group_idx = 0
		idx_in_group=0
		#line= pxml.addTextLine(reg,"ID"+str(group_idx+1))
		for n in order:
			word_idx = order.index(n)
			
			if idx_in_group>=groups[group_idx]:
				#line = pxml.selectNth('//_:TextLine',group_idx,reg)
				#line= pxml.selectNth(reg)
				group_idx+=1
				idx_in_group=0

			pxml.setProperty(words[n],'word_idx',str(word_idx))
			pxml.setProperty(words[n],"line",str(group_idx))
			pxml.moveElem(words[n],line)
			idx_in_group+=1

		# Write XML to file
		pxml.write('pagexmls/'+gtxml_name+".xml")
		cv2.imwrite(str(image_id)+'.jpg', img)


def draw_caption(image, box, caption):

	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def get_n_random_colors(n):
	colors = []
	for i in range(n):
		color = (int(255*np.random.random()),int(255*np.random.random()),int(255*np.random.random()))
		colors.append(color)
	return colors
def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--score_threshold', help='Score above which boxes are kept',default=0.5)
	parser.add_argument('--nms_threshold', help='Score above which boxes are kept',default=0.2)

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val,shuffle=False)

	retinanet = torch.load(parser.model)
	score_threshold = float(parser.score_threshold)
	nms_threshold = float(parser.score_threshold)
	use_gpu = True
	f = open('mAPs.txt','w')
	writer = csv.writer(f,delimiter = ",")
	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()
	thresholds = np.array([0.1 + 0.05*n for n in range(10)])
	for nms_threshold in thresholds:	
		for score_threshold in thresholds:
			mAPs = csv_eval.evaluate(dataset_val, retinanet,iou_threshold=0.5,score_threshold=score_threshold,nms_threshold = nms_threshold)	
			maps=np.mean([ap[0] for ap in mAPs.values()])
			writer.writerow([score_threshold,nms_threshold,maps])

if __name__ == '__main__':
 main()

