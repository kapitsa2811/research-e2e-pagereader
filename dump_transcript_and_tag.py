import pagexml
import os
import pdb
import csv
import re
def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)
GT_DIR = 'datasets/esposalles/data-esposalles/test'
re.sub("[^0-9]", "", "sdkjh987978asd098as0980a98sd")

dir_out = 'preds_txt'
if not os.path.exists(dir_out): os.mkdir(dir_out)

for f in sorted_nicely(os.listdir('pagexmls')):
	gt_xml = f[:24]
	gt_xml = os.path.join(GT_DIR,gt_xml+'.xml')
	pxml =pagexml.PageXML()
	
	pxml.loadXml(gt_xml)
	page = pxml.select('_:Page')
	page_id = pxml.getPropertyValue(page[0],key='idPage')
	record_id = f.split('.')[0][-2:]
	record_id = str(int(re.sub("[^0-9]", "", record_id))+1)

	pred_xml_file = f.split('.')[0]+'.xml'
	pred_xml = pagexml.PageXML()
	pred_xml.loadXml(os.path.join('pagexmls',pred_xml_file))
	words = pred_xml.select('//_:Word')
	fout = open(os.path.join(dir_out,"idPage"+page_id+"_Record"+record_id+"_output.txt"),"w")
	for i in range(len(words)):
		word = words[i]
		text = pxml.getTextEquiv(word)
		if len(text)<1: continue
		category = 'other'#pxml.getPropertyValue(word,key='category').split('_')[0]
		person = "_".join(pxml.getPropertyValue(word,key='category').split('_')[1:])
		fout.write(' '.join([text,"B-"+category])+'\n')
	fout.close()
