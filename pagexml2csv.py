#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import os
import pdb
import pagexml
import sys
import csv
import re
import argparse
import glob

#First parameter is the replacement, second parameter is your input string


def get_coords_and_transcript(pxml,textobject,key):
    regex = re.compile('[^a-zA-Z]')
    coords = pxml.getPoints(textobject)
    if len(coords)==4: 
        arg_max_coord=2
    else:
        arg_max_coord=1
    x0=int(coords[0].x)
    y0=int(coords[0].y)
    
    x1=int(coords[arg_max_coord].x)
    y1=int(coords[arg_max_coord].y)

    transcription = pxml.getTextEquiv(textobject)
    tag = pxml.getPropertyValue(textobject,key=key)
    line_transcript=[]
    w=transcription
    '''for w in transcription.split(" "):    
        if '<' in w:
            w = w.split('>')[1].split('<')[0]
    '''
    w= regex.sub('', w)
           
    line_transcript.append(w)
    line_transcript = " ".join(line_transcript)
    line_transcript = line_transcript.strip()
    return x0,y0,x1,y1,line_transcript,tag
    

def main(args=None):
    parser = argparse.ArgumentParser(description='Convert pagexml files to RetinaNet network csv groundtruth.')

    parser.add_argument('--pxml_dir', help='Path of directory with pagexml files.',default = ".")
    parser.add_argument('--fout', help='Path of gt file to be read by the model.',default = "train.csv")
    parser.add_argument('--classes_out', help='Path to save text category classes.')
    parser.add_argument('--seg_lev',help='segmentation level of the boxes to get (Word/TextLine)',default ="Word")
    parser.add_argument('--get_property',help='segmentation level of the boxes to get (Word/TextLine)',default =False)
    parser.add_argument('--property_key',help='key to get property from pagexml',default ='label')
    
    parser = parser.parse_args(args)
    pagexml.set_omnius_schema()
    pxml = pagexml.PageXML()
    if parser.classes_out is not None:
        classes_out = open(parser.classes_out,'w')
    csv_out = open(parser.fout,'w')
    writer = csv.writer(csv_out,delimiter=',')
    writer_classes = csv.writer(classes_out,delimiter=',')
    all_tags = []
    for root, dirs, files in os.walk(os.path.join(os.getcwd(),parser.pxml_dir)):
       for f in files:
        if '.xml' in f:
            pxml.loadXml(os.path.join(root,f))
            pages = pxml.select('_:Page')
            for page in pages:
                pagenum = pxml.getPageNumber(page)
                page_im_file =pxml.getPageImageFilename(page)
                page_im_file = os.path.join(os.getcwd(),root,page_im_file)    
                regions = pxml.select('_:TextRegion',page)
                for region in regions:
                    reg_tag=pxml.getPropertyValue(region,key=parser.property_key)
                    for textObject in pxml.select('_:'+parser.seg_lev,region):
                        x0,y0,x1,y1,transcription,tag=get_coords_and_transcript(pxml,textObject,parser.property_key)
                        if tag not in all_tags: all_tags.append(tag)
                        if x0>=x1 or y0>=y1: continue
                        if parser.get_property:
                            if len(tag)>0:
                                writer.writerow([page_im_file,x0,y0,x1,y1,tag,transcription])
                            else:

                                writer.writerow([page_im_file,x0,y0,x1,y1,reg_tag,transcription])
                        else:
                                writer.writerow([page_im_file,x0,y0,x1,y1,'text',transcription])


    if len(all_tags)>0:
        for idx,tag in enumerate(all_tags):
            writer_classes.writerow([tag,idx])

if __name__=='__main__':
    main()
