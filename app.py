from __future__ import absolute_import

from flask import Flask, request, send_file, current_app
from flask_cors import CORS, cross_origin
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
import json
import string
import random
import os
from io import BytesIO
import numpy as np
from PIL import Image
from urllib.request import urlretrieve
from facehandler import FaceHandler


facehandler = FaceHandler("mtcnninsigntface/mtcnn-model", "mtcnninsigntface/insightface-model/model", min_train = 5)


def http_mode():
    app = Flask(__name__)
    CORS(app)

    @app.route('/face_id', methods = ['GET', 'POST'])
    def faceID():
        if request.method == 'POST':
            if 'method' not in request.form:
                return json.dumps({'error': 'need a methd, either unlock or train'})

            method = request.form.get('method')
            if method == 'unlock':
                if 'image' not in request.files:
                    return json.dumps({'error': 'need a profile image'})

                image = request.files.get('image')
                try:
                    image = Image.open(BytesIO(image.read())).convert("RGB")
                except:
                    return json.dumps({'error': 'invalid profile image'})

                return facehandler.unlock(image)

            elif method == 'train':
                if 'image' not in request.files:
                    return json.dumps({'error': 'need some train images'})

                if 'user_id' not in request.form:
                    return json.dumps({'error': 'need an user_id'})

                
                image = request.files.get('image')
                try:
                    image = Image.open(BytesIO(image.read())).convert("RGB")
                except:
                    return json.dumps({'error': 'invalid profile image'})

                user_id = request.form.get('user_id')

                return facehandler.train(image, user_id)

            elif method == 'status':
                if 'user_id' not in request.form:
                    return json.dumps({'error': 'need an user_id'})

                user_id = request.form.get('user_id')

                return facehandler.check_status(user_id)


    # @app.route('/close_parsing', methods = ['GET', 'POST'])
    # def clothParse():
    #     if request.method == 'POST':
    #         if 'image' not in request.files:
    #             return json.dumps({'error': 'need an image'})

    #         image = request.files.get('image')
    #         try:
    #             img = Image.open(BytesIO(image.read())).convert("RGB")
    #         except:
    #             return json.dumps({'error': 'invalid image'})

    #         response = {}
    #         response['object_ids'] = [0,1]
    #         response['bboxes'] = [[0.9, 0.9, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4]]
    #         response['tag'] = ['pant', 'top']
    #         return json.dumps(response)



    @app.route('/health')
    def health():
        body = {
            'version': 1,
            'healthy': True
        }
        return json.dumps(body)

    app.run(host='0.0.0.0', debug=True, threaded=True, port= 5000)





if __name__ == '__main__':
    http_mode()


