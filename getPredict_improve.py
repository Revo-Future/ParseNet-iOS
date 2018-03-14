caffe_root = "/home/shaohu/SSD/deeplab-v2/"
import sys, getopt
sys.path.insert(0, caffe_root + 'python')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt
from skimage.measure import label

def readImageList(file_name):
        fin = file(file_name, 'r')
        lines = fin.readlines()
        imglist = []
        for line in lines:
                line = line.strip()
                imglist.append(line)
        return imglist


def resizeImage(image):
        width = image.shape[0]
        height = image.shape[1]
        maxDim = max(width,height)
        if maxDim>500:
            if height>width:
                ratio = float(500.0/height)
            else:
                ratio = float(500.0/width)
            image = PILImage.fromarray(np.uint8(image))
            image = image.resize((600, 600),resample=PILImage.BILINEAR)
            image = np.array(image)
        return image

def getpallete(num_cls):
        # this function is to get the colormap for visualizing the segmentation mask
        n = num_cls
        pallete = [0]*(n*3)
        for j in xrange(0,n):
                lab = j
                pallete[j*3+0] = 0
                pallete[j*3+1] = 0
                pallete[j*3+2] = 0
                i = 0
                while (lab > 0):
                        pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                        pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                        pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                        i = i + 1
                        lab >>= 3
        return pallete

def init_model(model_file, pretrained_file, gpudevice):
	if gpudevice >= 0:
                #Do you have GPU device? NO GPU is -1!
                has_gpu = 1
                #which gpu device is available?
                gpu_device=gpudevice#assume the first gpu device is available, e.g. Titan X
        else:
                has_gpu = 0
        if has_gpu==1:
                caffe.set_device(gpu_device)
                caffe.set_mode_gpu()
        else:
                caffe.set_mode_cpu()

        print "start..........."
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        print "end................"
	return net

def processing(inputfile):
	input_image = 255 * caffe.io.load_image(inputfile)
        input_image = resizeImage(input_image)

        width = input_image.shape[0]
        height = input_image.shape[1]

        image = PILImage.fromarray(np.uint8(input_image))
        image = np.array(image)
	mean_vec = np.array([104.00699, 116.66877, 122.67892], dtype=np.float32)
        reshaped_mean_vec = mean_vec.reshape(1, 1, 3)
        # Rearrange channels to form BGR
        im = image[:,:,::-1]
        # Subtract mean
        im = im - reshaped_mean_vec
        # Pad as necessary
        cur_h, cur_w, cur_c = im.shape
	cur_h, cur_w, cur_c = im.shape
        pad_h = 500 - cur_h
        pad_w = 500 - cur_w
        #im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
  	im = im*0.017
	return im, cur_h, cur_w

def parsenet_segmenter(net, inputs, inputfile):
	input_ = np.zeros((len(inputs),
        	600, 600, inputs[0].shape[2]),
        	dtype=np.float32)

    	for ix, in_ in enumerate(inputs):
        	input_[ix] = in_

   	 # Segment
    	caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                        dtype=np.float32)

    	for ix, in_ in enumerate(input_):
        	caffe_in[ix] = in_.transpose((2, 0, 1))

	print caffe_in.shape
	print "shape ..........."
    	out = net.forward_all(**{net.inputs[0]: caffe_in})
	print "ddddddddddddddddddddddd"
	print net.outputs[0]
	predictions = out[net.outputs[0]]
	print predictions[0].shape
	res = np.reshape(predictions[0], [predictions[0].shape[1], predictions[0].shape[2]])
        res = res.astype(np.uint8)
	return res

def run(net, inputfile, outputfile):
	[im, cur_h, cur_w] = processing(inputfile)
    	pallete = getpallete(256)
	segmentation  = parsenet_segmenter(net, [im], inputfile)
	print "segmentation",segmentation.shape
	segmentation2 = segmentation[0:cur_h, 0:cur_w]
	print "ddd"
    	output_im = PILImage.fromarray(segmentation2 , mode='L')
	print "mode",output_im.mode
	print "ddddddd"
    	output_im.putpalette(pallete)
    	output_im.save(outputfile)

#file_name = "/mogu/shaohu/jiepai_20170117/name_500.txt"
#file_name = "val_img.txt"
file_name = "name.txt"
MODEL_FILE = 'train_parse_bn.prototxt'
PRETRAINED = 'mobile_iter_160000.caffemodel'
gpudevice = 1
imglist = readImageList(file_name)
net = init_model(MODEL_FILE, PRETRAINED, gpudevice)
for line in imglist:
	line = line.strip()
	print line
 	outputfile = './' + os.path.basename(line)+"_deeplab.png"
	run(net, line, outputfile)
