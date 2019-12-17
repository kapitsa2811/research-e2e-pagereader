# End-to-End Handwritten Text Detection andTranscription in Full Pages

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

## Training


```
./experiment_e2e.sh
```

