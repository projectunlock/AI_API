from io import BytesIO
from PIL import Image
import numpy as np
import json
from flask import request, Response, send_file, make_response
import json, os, sys, urllib.request
from datetime import datetime
from urllib.parse import urlparse
import threading
import boto3
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class Utility(object):
    # def url_to_cv_image(url):
    #     return cv2.imdecode(np.asarray(bytearray(urllib.request.urlopen(url).read()), dtype="uint8"), cv2.IMREAD_COLOR)

    # def file_to_cv_image(file):
    #     return cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    def __init__(self):
        CONFIG = {
            'BUCKET_REGION_MAP':{
                'adobesearch-dev-uw2-redpill': 'us-west-2',
                'adobe-stock-search-uw2': 'us-west-2'
            }
        }
        self._s3_clients = {}
        for bucket, region in CONFIG['BUCKET_REGION_MAP'].items():
            client = boto3.client('s3', region_name=region)
            self._s3_clients[bucket] = client
        self._s3_lock = threading.Lock()

    def get_image(self, url=None, file=None):
        if url:
            return self.url_to_pillow_img_with_validation(url)
        elif file:
            return self.file_to_pillow_img_with_validation(file)
        else:
            return None

    def get_video(self, file=None):
        if file:
            if not self.is_valid_video_name(file.filename):
                return None
            else:
                video_filename = self.generate_filename(file.filename.rsplit('.', 1)[1].lower())
                file.save(video_filename)
                return video_filename
        else:
            return None

    def url_to_pillow_img_with_validation(self, url):
        if not self.is_valid_img_name(url):
            return None
        try:
            scheme, bucket, path, *_ = urlparse(url)
            if scheme in ('http', 'https'):
                req = urllib.request.Request(url, headers=self.get_header())
                body = urllib.request.urlopen(req)
                img = Image.open(BytesIO(body.read()))
                return self.load_rgb_image(img)
            elif scheme == 's3':
                client = self._s3_clients.get(bucket)
                if client is None:
                    self._s3_lock.acquire()
                    try:
                        region = boto3.client('s3').get_bucket_location(Bucket=bucket)['LocationConstraint']
                        client = boto3.client('s3', region_name=region)
                        self._s3_clients[bucket] = client
                    except:
                        raise
                    finally:
                        self._s3_lock.release()
                body = client.get_object(Bucket=bucket, Key=path[1:]).get('Body')
                img = Image.open(BytesIO(body.read()))
                return self.load_rgb_image(img)
            else:
                return None
        except:
            return None
        
    def file_to_pillow_img_with_validation(self, file):
        if not self.is_valid_img_name(file.filename):
            return None
        try:
            img = Image.open(BytesIO(file.read()))
            return self.load_rgb_image(img)
        except:
            return None

    def pillow_img_to_buffer(self, img):
        byte_io = BytesIO()
        img.save(byte_io, 'JPEG')
        byte_io.seek(0)
        return byte_io


    def pillow_img_to_file(self, img):
        if img != None:
            image_filename = self.generate_filename('jpg')
            img.save(image_filename, "JPEG")
            return image_filename

    def is_valid_img_name(self, name):
        return '.' in name and name.rsplit('.', 1)[1].lower() in ["jpg", "png", "gif", "tif", "jpeg"]

    def is_valid_video_name(self, name):
        return '.' in name and name.rsplit('.', 1)[1].lower() in ["mp4", "avi", "mov", "mkv"]
    
    def generate_filename(self, ext):
        return os.path.join('/tmp', datetime.now().strftime("%Y%m%d_%H%M%S%f")+'.'+ext)

    def get_header(self):
        hdr = { 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'}
        return hdr

    def load_rgb_image(self, img):
        if not img:
            return None
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if hasattr(img, '_getexif'):
            try:
                exif = img._getexif()
                if exif:
                    rotation = {
                        3: 180,
                        6: 270,
                        8: 90
                    }.get(exif.get(274))
                    if rotation:
                        img = img.rotate(rotation, expand=True)
            except:
                pass
        return img

    def sub_voting(self, voter, pillow_img, stock):
        img_file = self.pillow_img_to_buffer(pillow_img)
        stats = voter.unify_vote(paras={}, modules=['all'], stock=stock, sim_size=75, bucket=20, img_file=img_file)
        return stats

    def resp_failed_body(self, msg):
        resp = Response(msg, status=400)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    def resp_json_body(self, stats):
        resp = Response(json.dumps(stats, sort_keys=True))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    def resp_file_body(self, img_io):
        resp = make_response(send_file(img_io, mimetype='image/jpeg'))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
