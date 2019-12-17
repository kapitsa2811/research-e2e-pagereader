# End-to-End Handwritten Text Detection and Transcription in Full Pages

![img3](http://www.cvc.uab.es/people/mcarbonell/images/page.jpg)

Pytorch  implementation of the paper [End-To-End Handwritten Text Detection and Transcription in Full Pages](http://www.cvc.uab.es/people/mcarbonell/papers/wml.pdf) by Manuel Carbonell, Joan Mas, Mauricio Villegas, Alicia Fornés and Josep Lladós.


## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip3 install cffi

pip3 install cython

pip3 install opencv-python

pip3 install requests

pip3 install editdistance
```
Warp ctc pytorch: https://github.com/SeanNaren/warp-ctc

Python pagexml: https://github.com/omni-us/pagexml/tree/master/py-pagexml

## Data
Train, validation and test examples's ground truth boxes and text contents must be listed in a csv file with the following format:
```
path/to/image1.png,text_bbox_1_x0,text_bbox_1_y0,text_bbox_1_x1,text_bbox_1_y1,'text',text_bbox_1_content
path/to/image1.png,text_bbox_2_x0,text_bbox_2_y0,text_bbox_2_x1,text_bbox_2_y1,'text',text_bbox_2_content
...
path/to/imageN.png,text_bbox_1_x0,text_bbox_1_y0,text_bbox_1_x1,text_bbox_1_y1,'text',text_bbox_1_content

```

## Training


```
./experiment_e2e.sh
```

