import pdb
import csv
import sys
import re
import numpy as np
import string 
import os
import pdb
regex = re.compile('[^a-zA-Z]')
MAX_STRING_LENGTH = 20
PAGE_W = 2479
PAGE_H = 3542
FILE_IN = sys.argv[1]
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "
file_dir = os.path.dirname(os.path.realpath(__file__))
d = dict(zip(string.ascii_lowercase, range(0,27)))

def string_to_ids(string):
    string  = string.lower()
    regex.sub('', string)
    ids_out = np.zeros(MAX_STRING_LENGTH)
    ids = []
    for letter in string:
       ids.append(str(d.get(letter," "+str(len(alphabet)-1)+" ")))
    for i in range(MAX_STRING_LENGTH-len(string)):
	ids.append(" 27 ")
    ids_out=" ".join(ids)
    #print string,ids_out
    return " ".join(ids)

f_in = open(FILE_IN,'r')

lines = f_in.readlines()

previous_line_form=""
grammatical_tags = []
hwratio =0
fileout_train =open('train_words.csv','w')
fileout_valid =open('valid_words.csv','w')
fileout_test =open('test_words.csv','w')

writer_train = csv.writer(fileout_train,delimiter = ',')
writer_valid = csv.writer(fileout_valid,delimiter = ',')
writer_test = csv.writer(fileout_test,delimiter = ',')

train_forms = open('tr.lst','r')
train_forms =train_forms.readlines() 
train_forms = [f.strip() for f in train_forms]

val1_forms=open('va.lst','r')
val1_forms = val1_forms.readlines()
val1_forms = [f.strip() for f in val1_forms]


test_forms=open('te.lst','r')
test_forms = test_forms.readlines()
test_forms = [f.strip() for f in test_forms]

not_found=0
total=0
for line in lines:
	if '#' in line: continue
	cols = line.split(" ")
	current_line_form = '-'.join(cols[0].split('-')[0:2])
	if current_line_form<>previous_line_form:
		write_valid=(np.random.random()<0.2)
	image_path = os.path.join(file_dir,'forms',current_line_form+'.png')
	x0 = int(cols[3])
	y0 = int(cols[4])
	w = int(cols[5])
	h = int(cols[6])
	x1 = x0+w
	y1 = y0+h
	clean_text=[]
	total+=1

	box_transcription =cols[-1].strip() 
	for w in box_transcription.split('|'):
		clean_w=""
		for letter in w.lower():
			if letter not in alphabet:
				continue
			else:
				clean_w=clean_w+letter
		clean_text.append(clean_w)
	box_transcription = " ".join(clean_text)
	grammatical_tag = cols[-2][:2]
	if grammatical_tag not in grammatical_tags:
		grammatical_tags.append(grammatical_tag)
	#box_transcription_ids = string_to_ids(box_transcription)
	if len(box_transcription)>0 and w>0 and h>0 and x0>=0 and y0>=0:
		if current_line_form in val1_forms:
			writer_valid.writerow([image_path,x0,y0,x1,y1,'text',box_transcription])
		elif current_line_form in train_forms:
			writer_train.writerow([image_path,x0,y0,x1,y1,'text',box_transcription])
		elif current_line_form in test_forms:
			writer_test.writerow([image_path,x0,y0,x1,y1,'text',box_transcription])
		else:
			not_found+=1
			print "FORM NOT FOUND",current_line_form,cols[1]
			continue
	#fileout.write("BB "+str(x0)+" "+str(y0)+" "+str(x1)+" "+str(y1)+" "+str(x0)+" "+str(y1)+" "+str(x1)+" "+str(y1)+" "+str(h)+" "+str(h)+" "+grammatical_tag+' '+str(len(box_transcription))+" "+box_transcription_ids+ "\n")
	previous_line_form = current_line_form
print len(lines)
print grammatical_tags
print float(not_found)/total
