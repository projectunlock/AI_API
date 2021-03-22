from __future__ import absolute_import

import sys, os, io, random
import numpy as np
from base64 import encodebytes
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from collections import defaultdict
from datetime import datetime


sys.path.append("modanet")
from mrcnn import utils
import mrcnn.model as modellib
import custom
from collections import defaultdict


class InferenceConfig(custom.CustomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ClothParser(object):

    def __init__(self, model_dir = "./modanet"):
        config = InferenceConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
        MODEL_PATH = os.path.join(model_dir, "mask_rcnn_Modanet.h5")
        model.load_weights(MODEL_PATH, by_name=True)
        self.model = model
        self.class_names = ['BG','bag','belt','boots','footwear','outer','dress','sunglasses','pants','top','shorts','skirt','headwear','scarf/tie']
        self.cloth_path = "clothes.npy"
        self.saved_clothes = defaultdict(dict)
        if os.path.exists(self.cloth_path):
            self.saved_clothes = np.load(self.cloth_path, allow_pickle = True).item()
        self.cloth_save_dir = "clothes"
        if not os.path.exists(self.cloth_save_dir):
            os.mkdir(self.cloth_save_dir)

        print ("loaded cloth parse model")
        print ("cloth dict status:")
        print (self.saved_clothes.items())



    def parse(self, ori_image, user_id, threshold = 0 ):
        # pass in  PIL image as RGB
        image = np.array(ori_image)
        result = self.model.detect([image], verbose=0)[0]
        bboxes = result['rois']
        class_ids = result['class_ids']
        scores = result['scores']

        cutoff = scores > threshold
        scores = scores[cutoff]
        bboxes = bboxes[cutoff]
        class_ids = class_ids[cutoff]
        labels = [self.class_names[class_id] for class_id in class_ids]
        
        filename = self.save_info(bboxes, scores, labels, user_id, ori_image)

        return filename




    def search(self, tag, user_id):
        # now only support keyword search
        if tag not in self.class_names[1:]:
            return {'error': "only support search in " + '; '.join(self.class_names[1:])}

        if tag not in self.saved_clothes[user_id]:
            return {'error': "%s not found for %s" %(tag, user_id)}

        clothes = self.saved_clothes[user_id][tag]
        result = []
        if len(clothes) > 0:
            for item in clothes:
                _, _, filename = item
                result.append(self.encode(filename))

        result = {'result': result}
        return result


    def save_info(self, bboxes, scores, labels, user_id, pil_image):
        filename = os.path.join(self.cloth_save_dir, self.generate_filename(user_id, 'jpg'))
        draw = ImageDraw.Draw(pil_image)
        
        for i, label in enumerate(labels):
            if label not in self.saved_clothes[user_id]:
                self.saved_clothes[user_id][label] = []
                #self.saved_clothes[user_id] = {label: []}
                print ("created %s for %s" %(label, user_id))
                print (self.saved_clothes[user_id][label])
            box = bboxes[i]
            y1, x1, y2, x2 = box
            draw.rectangle(((x1, y1), (x2, y2)), outline = (0,0,0))

            color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            draw.text((x1 + 5, y1 + 5), label, fill = color)
            self.saved_clothes[user_id][label].append([box, scores[i], filename])
        print (self.saved_clothes[user_id])

        pil_image.save(filename)
        np.save(self.cloth_path, self.saved_clothes)
        return filename



    def generate_filename(self, user_id, ext):
        return user_id + "_"+ datetime.now().strftime("%Y%m%d_%H%M%S%f")+'.'+ext

    def encode(self, image_path):
        pil_img = Image.open(image_path, mode='r') # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        return encoded_img


