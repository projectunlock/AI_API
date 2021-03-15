#
# ADOBE CONFIDENTIAL
# __________________
#
# Copyright 2016 Adobe Systems Incorporated
# All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of
# Adobe Systems Incorporated and its suppliers, if any. The intellectual
# and technical concepts contained herein are proprietary to Adobe Systems
# Incorporated and its suppliers and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law. Dissemination of this information or reproduction of this
# material is strictly forbidden unless prior written permission is obtained
# from Adobe Systems Incorporated.
#


import sys
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import math, os
from glob import glob
import sklearn
import sklearn.preprocessing
import cv2
import face_preprocess
import mxnet as mx
from mtcnn_detector import MtcnnDetector


class MtcnnInsightface(object):

    class Face(object):
        def __init__(self, box, desc, score):
            self.box=box
            self.desc = desc
            self.score=score

    def __init__(self, detector_dir, recognize_dir, mx_epoch = 0, image_size = [112,112], layer = 'stage4_unit3_bn3', gpu = -1):
        os.environ['GLOG_minloglevel'] = '2'
        self.det = 0
        if gpu >=0:
            self.ctx = mx.gpu(gpu)
        else:
            self.ctx = mx.cpu()
        self.det_threshold = [0.6,0.7,0.8]
        
        self.mt_detector = MtcnnDetector(model_folder= detector_dir, ctx=self.ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
        sym, arg_params, aux_params = mx.model.load_checkpoint(recognize_dir, mx_epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output']

        rec_model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
        rec_model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        rec_model.set_params(arg_params, aux_params)
        self._recognition_model = rec_model
        print ("loaded detection and recognition model successfully")
        


    def face_detection(self, face_img):
        if isinstance(face_img, str):
            face_img = cv2.imread(face_img)
        ret = self.mt_detector.detect_face(face_img, det_type = self.det)
        return ret


    def face_recognition(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = np.transpose(img, (2,0,1))    
        input_blob = np.expand_dims(img, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self._recognition_model.forward(db, is_train=False)
        embedding = self._recognition_model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(np.mean(embedding, axis=(2,3))).flatten()
        return embedding

    def face_alignment(self, image, landmarks, desiredLeftEye = [0.35, 0.35], scale = 1):
        landmarks = landmarks.astype(np.float32)
        leftEye = landmarks[0]
        rightEye = landmarks[1]
        dY = rightEye[1] - leftEye[1]
        dX = rightEye[0] - leftEye[0]
        angle = np.degrees(np.arctan2(dY, dX)) #- 180
        
        
        height, width = image.shape[:2]
        desiredFaceWidth = min(width, int(abs(dX) * 4))
        desiredFaceHeight = min(height, 4*int(landmarks[-1][1] - min(leftEye[1], rightEye[1])))
        
        eyesCenter = ((leftEye[0] + rightEye[0]) // 2, (leftEye[1] + rightEye[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)
        
        return output


    def extract_feat(self, ori_img, save_prefix = None, bodyskin_prefix = None):
        ori_img = cv2.imread(ori_img)
        #ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        if bodyskin_prefix:
            if os.path.exists(bodyskin_prefix + "skin.png") and os.path.exists(bodyskin_prefix + "mask.png"):
                skin_img = cv2.imread(bodyskin_prefix + "skin.png")
                mask_img = cv2.imread(bodyskin_prefix + "mask.png")
            else:
                print ("no valid bodyskin_prefix: " + bodyskin_prefix)
                return


        ret = self.face_detection(ori_img)
        all_faces, crop_bbs, bboxes = [], [], []
        if ret:
            bboxes, points = ret
            for i in range(len(bboxes)):
                box = bboxes[i]
                landmarks = points[i,:].reshape((2,5)).T
                face_img, M = face_preprocess.preprocess(ori_img, box, landmarks, image_size='112,112')
                face = self.Face(box, self.face_recognition(face_img), box[-1])
                all_faces.append(face)

                # save crops
                bb = np.zeros(4, dtype=np.int32)
                margin = max(abs(box[0] - box[2]), abs(box[1] - box[3]))/1.2
                bb[0] = np.maximum(box[0]-margin/2, 0)
                bb[1] = np.maximum(box[1]-margin/2, 0)
                bb[2] = np.minimum(box[2]+margin/2, ori_img.shape[1])
                bb[3] = np.minimum(box[3]+margin/2, ori_img.shape[0])
                face_crop = ori_img[bb[1]:bb[3],bb[0]:bb[2],:]

                # align_landmarks = landmarks
                # align_landmarks[:,0] -= bb[0]
                # align_landmarks[:,1] -= bb[1]
                # face_crop_aligned = self.face_alignment(face_crop, align_landmarks)

                if save_prefix:
                    cv2.imwrite(save_prefix + str(i)+'_crop.png', face_crop)   
                    cv2.imwrite(save_prefix + str(i)+'_crop_aligned.png', face_img)  

                if bodyskin_prefix:
                    face_aligned_skin = cv2.warpAffine(skin_img,M,(112,112), borderValue = 0.0)
                    face_aligned_mask = cv2.warpAffine(mask_img,M,(112,112), borderValue = 0.0)
                    face_aligned_mask = cv2.cvtColor(face_aligned_mask, cv2.COLOR_BGR2GRAY)

                    cv2.imwrite(save_prefix + str(i)+'_crop_aligned_skin.png', face_aligned_skin)   
                    cv2.imwrite(save_prefix + str(i)+'_crop_aligned_mask.png', face_aligned_mask)   
                    

        else:
            print ("no detection from mtcnn model")


        return all_faces, bboxes