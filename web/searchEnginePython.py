# -*- coding: utf-8 -*-
import cherrypy
import pickle
import urllib
import os
import numpy
from scipy import io
import numpy as np
import h5py
import sys

sys.path.append('../image_retrieval')
from keras.applications.vgg16 import VGG16
from extract_cnn_vgg16_keras import extract_feat
from keras.preprocessing import image
from numpy import linalg as LA

import threading

"""
This is the image search demo.
"""


class SearchDemo:

    def __init__(self):
        # load list of images
        self.path = './thumbnails/'
        #self.path = './images/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)

        #input_shape = (224, 224, 3)
        #self.model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)

        # set max number of results to show
	h5f = h5py.File('../image_retrieval/featureCNN.h5','r')
        self.feat = h5f['dataset_1'][:]
	self.imNameList = h5f['dataset_2'][:]
	h5f.close() 
        self.imNum, self.dim = self.feat.shape
        self.maxres = 54

	self.mutex = threading.Lock()

        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>以图搜图</title>
            <link rel="stylesheet" href="/bootstrap/css/bootstrap.min.css">
            <link rel="stylesheet" href="style.css">
            <script src='http://cdn.bootcss.com/jquery/1.11.2/jquery.min.js'></script>
            <script src='/bootstrap/js/bootstrap.min.js'></script>
	    <!-- file input --> 
	    <link href="bootstrap/css/fileinput.min.css" rel="stylesheet"> 
	    <script src="bootstrap/js/fileinput.min.js"></script> 
            </head>
            <body>
            """
        self.footer = """
            </html>
            """

    @cherrypy.expose
    def shutdown(self):  
        cherrypy.engine.exit()

    def index(self, query=None):

        html = self.header
        html += """
                <div class="jumbotron">
                    <div class="row">
                        <div class="col-lg-4 col-lg-offset-4">
                            <div class="input-group input-group-lg">
                                <input type="text" class="form-control" placeholder="以图搜图" name="srch-term" id="srch-term">
                                    <div class="input-group-btn">
                                        <button class="btn btn-gray" type="submit"><i class="glyphicon glyphicon-search"></i></button>
                                    </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="randButton">
                    <div class="row">
                        <button class="btn btn-default btn-lg btn-success" type="button"><a href='?query='>随机图片</a></button>
                        <!--<button class="btn btn-default btn-lg btn-warning" type="button"><a id="shutdown"; href="./shutdown">关闭服务器</a></button>-->
                    </div>
		    <!--
		    <div>
		    <form>
		      <input type="file" id="avatar" name="avatar">
			<button class="btn btn-default btn-lg btn-success" type="button"><a href='?upload='>上传图片</a></button>
		    </form>
		    </div>
		    -->
                </div>
            """

        if query:
            #print "query",query
	    #query_file= 'thumbnails/' + query.split("/")[-1]
	    query_file = query

	    print query_file
      	    input_shape = (224, 224, 3)
	    self.mutex.acquire()
            model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)
	    self.mutex.release()
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

            for imname in imlist:
	        imname='db_shoes/'+imname;
                html += "<div class='col-xs-6 col-sm-4 col-md-2 marginDown' >"

                html += "<a href='?query=db/"+imname+"' class='thumbnail'>"

                html += "<img class='img-responsive' style='max-height:220px' src='"+imname+"'  />"
                html += "</a>"
                html += "</div>"
        else:
            # show random selection if no query
            numpy.random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<div class='col-xs-6 col-sm-4 col-md-2 marginDown' >"
                #html += "<div class='col-sm-6 col-md-3'>"

                html += "<a href='?query="+imname+"' class='thumbnail'>"
                #html += "<a href='?query="+imname+"'>"

                html += "<img class='img-responsive' style='max-height:200px' src='"+imname+"'  />"
                #html += "<img src='"+imname+"' width='200px' height='200px' />"
                html += "</a>"
                html += "</div>"
                html += "</body>"
        html += self.footer
        html += """
                <footer>
                <p class='linkings'><a href='http://yongyuan.name/'></a></p>
                </footer>
                """
        return html

    index.exposed = True

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
