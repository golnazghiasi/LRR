## Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation
This repository contains code and models for the method described in:

[Golnaz Ghiasi, Charless C. Fowlkes, "Laplacian Pyramid Reconstruction and
Refinement for Semantic Segmentation", ECCV 2016](http://arxiv.org/abs/1605.02264)


This library is written in Matlab, it uses [Matconvnet](https://github.com/vlfeat/matconvnet)
library and is based on the following work:

* [matconvnet-fcn](https://github.com/vlfeat/matconvnet-fcn)

---- 

This code is tested on Linux using matconvnet v1.0-beta20 and cuDNN 5

### Testing pre-trained model on PASCAL VOC and MS COCO data on PASCAL VOC validation data
Download [pre-trained models](http://www.ics.uci.edu/~gghiasi/papers/LRR/models.tar.gz) and
extract it into models directory.

Specify matconvnet path in "LRRTestOnPascal.m" and execute it.

### Training LRR
comming soon

### Issues, Questions,  etc
Please contact "gghiasi @ ics.uci.edu"

---- 

**Copyright (C) 2016 Golnaz Ghiasi, Charless C. Fowlkes**

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

