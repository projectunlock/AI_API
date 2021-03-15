import os, sys, glob, csv
import numpy as np
from PIL import Image
import cv2
sys.path.append(os.path.join(os.path.dirname('__file__'), 'src'))
sys.path.append("/home/ubuntu/bodyskin/src/")

from utility import Utility
from model import MtcnnInsightface
#from bodyskin import BodySkin



# occlusion_file = "/home/ubuntu/demo/app/src/a1442190.csv"
# cids = []
# with open(occlusion_file) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         cid = row[-4]
#         if cid == 'cid':
#             continue
#         occlusion_score = int(row[5])
#         if occlusion_score == 0:
#             cids.append(cid)
# print (len(cids))


def detect(img_file):
    face_model = MtcnnInsightface('mtcnn-model/', 'insightface-model/model')
    try:
        cv2.imread(img_file)
    except:
        print ("failed process img to cv2 for " + img_file)
        exit(0)

    try:
        _, bboxes = face_model.extract_feat(img_file, save_prefix = img_file.split('.')[0] + "_")
    except:
        print ("error in face detection")
        exit(0)

    print ("detected %s bboxes" %(len(bboxes)))





def sequential_bbox_process(folder = '/home/ubuntu/30k'):
    face_model = MtcnnInsightface('mtcnn-model/', 'insightface-model/model')
    count = 0
    save_npy = {}
    for img_file in glob.glob(folder + "/*"):
        cid = img_file.split('/')[-1].split('.')[0]
        #if cid in cids:
        try:
            cv2.imread(img_file)
        except:
            print ("failed process img to cv2 for " + img_file)
            continue

        try:
            _, bboxes = face_model.extract_feat(img_file)
            save_npy[cid] = bboxes
            count +=1
        except:
            continue

        if count % 1000 == 0 and count > 0:
            print ("processed "+str(count))
            np.save("bboxes.npy", save_npy)

    print ("obtained a total of %s faces" %(count))
    np.save("bboxes.npy", save_npy)



def sequential_crop_process(folder = '/home/ubuntu/30k', occlusion_file = False, withBox = True):
    face_model = MtcnnInsightface('mtcnn-model/', 'insightface-model/model', gpu = 0)
    #bodyskin_model = BodySkin("/home/ubuntu/bodyskin/models/", True)
    count = 0
    if withBox:
        save_npy = {}
    for img_file in glob.glob(folder + "/*"):
        cid = img_file.split('/')[-1].split('.')[0]
        if occlusion_file:
            if cid not in cids:
                continue

        try:
            cv2.imread(img_file)
        except:
            print ("failed process img to cv2 for " + img_file)
            continue

        try:
            #bodyskin_model.predict(img_file, save_prefix = '/home/ubuntu/bodyskin/results/' + cid)
            _, bboxes = face_model.extract_feat(img_file, save_prefix = '/home/ubuntu/mtcnninsigntface/crops/'+ cid + '_', \
                bodyskin_prefix = '/home/ubuntu/bodyskin/results/' + cid + '_')
            count +=1
            if withBox:
                save_npy[cid] = bboxes
        except:
            continue
    
        if count % 1000 == 0:
            print ("processed "+str(count))
            if withBox: 
                np.save("bboxes.npy", save_npy)

    if withBox: np.save("bboxes.npy", save_npy)
    print ("obtained a total of %s faces" %(count))


def sequential_feat_process(layer, folder = '/home/ubuntu/30k'):
    face_model = MtcnnInsightface('mtcnn-model/', 'insightface-model/model', layer = layer)
    records = {}
    count = 0
    for img_file in glob.glob(folder + "/*"):
        # for 30k full res
        # items = img_file.split('/')[-1].split('_')
        # cid = items[1]
        # id32 = items[2]

        # for 30k
        cid = img_file.split('/')[-1].split('.')[0]
        try:
            img = Image.open(img_file)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except:
            print ("failed process img to cv2 for " + img_file)
            continue

        try:
            face = face_model.extract_feat(img)[0][0]
        except:
            continue
        
        if face:
            count +=1
            records[cid] = face
    
        if count % 1000 == 0:
            print ("processed "+str(count))
            np.save("faces_"+layer+".npy", records)
            # if count % 1e4 == 0:
            #     np.save("faces_%s.npy" %(count), records)
            #     records = {}


    print ("obtained a total of %s faces" %(count))
    np.save("faces_"+layer+".npy", records)


def knn(img_file, feats, cids):
    img_feat = face_model.extract_feat(cv2.imread(img_file))[0].desc
    feats = np.array(feats)
    diff = np.linalg.norm(feats - img_feat, axis = 1)
    tups = list(zip(diff, cids))
    tups.sort()
    _, sorted_cids = zip(*tups)
    return sorted_cids


if __name__ == '__main__':
    args = sys.argv
    command = args[1]
    if command == 'detect':
        file = args[2]
        detect(file)
    if command == 'crop':
        sequential_crop_process()

    elif command == 'feat':
        layer = args[2]
        sequential_feat_process(layer)

    elif command == 'box':
        sequential_bbox_process()

    elif command == 'knn':
        data_file = '/home/ubuntu/mona/feat/faces.npy'
        feats, cids = [], []
        
        dic = np.load(data_file, allow_pickle = True).item()
        for cid in dic:
            feats.append(dic[cid].desc)
            cids.append(cid)
        print ("total %s cids" % len(cids))

        img_file = args[2]
        knn_cids = knn(img_file, feats, cids)
        print (knn_cids)


