#!/usr/bin/env python
#########################################################################
# Author: lincolnlin
# Created Time: Sun Aug 13 08:25:53 2017
# File Name: test2.py
# Description: 
#########################################################################

import itchat, time
from itchat.content import *

import numpy
#from numpy import *
from scipy import io
import numpy as np
import h5py
import sys

sys.path.append('../image_retrieval')
sys.path.append('../object_detect/keras_frcnn/')
import frcnn_tool 

from keras.applications.vgg16 import VGG16
from extract_cnn_vgg16_keras import extract_feat
from keras.preprocessing import image
from numpy import linalg as LA
import os

class SearchEngine:

    def __init__(self):
	h5f = h5py.File('../image_retrieval/featureCNN.h5','r')
        self.feat = h5f['dataset_1'][:]
	self.imNameList = h5f['dataset_2'][:]
	h5f.close() 
        self.imNum, self.dim = self.feat.shape
        self.maxres = 54
	print "init SearchEngine"

    def query(self, query_file):
	print query_file
      	input_shape = (224, 224, 3)
        model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)
	queryVec = extract_feat(model,query_file)

	scores = np.dot(queryVec, self.feat.T)
	rank_ID = np.argsort(scores)[::-1]
	rank_score = scores[rank_ID]
	print rank_ID
	print rank_score

	# number of top retrieved images to show
	maxres = 3
	imlist = [self.imNameList[index] for i,index in enumerate(rank_ID[0:maxres])]
	print "top %d images in order are: " %maxres, imlist

	result=[]
        for imname in imlist:
	    imname='../db/db_shoes/'+imname;
	    result.append(imname)
	return result

engine=SearchEngine()

@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    msg.user.send('%s: %s' % (msg.type, msg.text))

@itchat.msg_register([PICTURE, RECORDING, ATTACHMENT, VIDEO])
def download_files(msg):
    msg.download(msg.fileName)
    typeSymbol = {
        PICTURE: 'img',
        VIDEO: 'vid', }.get(msg.type, 'fil')
    print typeSymbol
    print msg.fileName

    #frcnn
    #os.system("cd ../object_detect/frcnn && python test_frcnn.py -p ../"+msg.fileName+"  && cd -") 
    #itchat.send_image("frcnn/results_imgs/0.png",toUserName=msg.fromUserName)
    frcnn_tool.frcnn_box(msg.fileName,"box.jpg")
    itchat.send_image("box.jpg",toUserName=msg.fromUserName)
    
    #index query
    #result=engine.query(msg.fileName)
    #for img in result:
    	#itchat.send_image(img,toUserName=msg.fromUserName)
    return None
    #return '@%s@%s' % (typeSymbol, result[0])

@itchat.msg_register(FRIENDS)
def add_friend(msg):
    msg.user.verify()
    msg.user.send('Nice to meet you!')

@itchat.msg_register(TEXT, isGroupChat=True)
def text_reply(msg):
    if msg.isAt:
        msg.user.send(u'@%s\u2005I received: %s' % (
            msg.actualNickName, msg.text))

#itchat.auto_login(enableCmdQR =1, hotReload=True)
itchat.auto_login(enableCmdQR =1)
itchat.run(True)
