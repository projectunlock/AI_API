from __future__ import absolute_import

import sys, os
import numpy as np
from PIL import Image
from collections import defaultdict

sys.path.append("mtcnninsigntface")

from model import MtcnnInsightface

class FaceHandler(object):

    def __init__(self, detector_dir, recognize_dir, gpu = -1, min_train = 5):
        self.mtcnninsightface = MtcnnInsightface(detector_dir, recognize_dir, gpu = gpu)
        self.faces_path = "faces.npy"
        self.min_train = min_train
        self.status = defaultdict(int)
        if os.path.exists(self.faces_path):
            saved_faces = np.load(self.faces_path, allow_pickle = True).item()
            for user_id in saved_faces:
                self.status[user_id] = len(saved_faces[user_id])

        print ("status:")
        print (self.status.items())


    def train(self, image, user_id):
        if os.path.exists(self.faces_path):
            saved_faces = np.load(self.faces_path, allow_pickle = True).item()
        else:
            saved_faces = defaultdict(list)

        # write the new image feature to local
        image = np.array(image)[:,:,::-1]
        features, _ = self.mtcnninsightface.extract_feat(image, save_folder = os.path.join("/demo/faces", user_id))
        if len(features) < 1:
            return {'error': 'there is no face in the input image'}

        saved_faces[user_id].append(features[0])
        count = len(saved_faces[user_id])
        np.save(self.faces_path, saved_faces)
        self.status[user_id] = count

        # check train status
        if count < self.min_train:
            return {'status': 'face id need at least %s faces to train, now have %s for %s' %(self.min_train, count, user_id)}
        else:
            return {'status': 'face id ready to use'}

    def check_status(self, user_id):

        count = self.status[user_id]
        if count < self.min_train:
            status = {'status': 'face id need at least %s faces to train, now have %s for %s' %(self.min_train, count, user_id)}
        else:
            status = {'status': 'face id ready to use for %s' %(user_id)}

        return status

    def unlock(self, image, threshold = 1):
        if os.path.exists(self.faces_path):
            saved_faces = np.load(self.faces_path, allow_pickle = True).item()
            min_distance = np.inf
            closest_id = None
            image = np.array(image)[:,:,::-1]
            features, _ = self.mtcnninsightface.extract_feat(image)
            if len(features) < 1:
                return {'error': 'there is no face in the image'}
            feature = features[0]

            print ("diff")
            for user_id, cluster_features in saved_faces.items():
                diff = min(np.linalg.norm(np.array(cluster_features) - feature, axis = 1))
                if diff < min_distance:
                    min_distance = diff
                    closest_id = user_id
                    print (str(diff) + " with " + user_id)

            if threshold:
                if min_distance >= threshold:
                    return {'status': 'face id unlock fails'}

            print ("unlock with user_id " + closest_id)
            if self.status[closest_id] >= self.min_train:
                return {'status': closest_id}
            else:
                return {'status': 'face id unlock fails'}
        
        return {'status': 'face id need at least %s faces to train, now have nothing ' %(self.min_train)}



        




