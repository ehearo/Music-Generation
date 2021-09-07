import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import streamlit as st
import matplotlib.pyplot as plt
from skimage import transform as trans
from sklearn.preprocessing import normalize
from retinaface import RetinaFace

def aligned_face(img, landmark):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
    dst = np.array(landmark, dtype=np.float32).reshape(5, 2)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue = 0)
    return aligned

def face_detection(argv, filename):
    #load Image
    img_path = argv
    file_name = filename
    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float32')
    
    # init with normal accuracy option
    detector = RetinaFace(quality="normal")
    facePositions = detector.predict(img)
    
    output_path = os.path.dirname(__file__) + '/detect_face/'
    output_path_gray = os.path.dirname(__file__) + '/gray_color/'
    
    if len(facePositions) > 0:

        for facePosition in facePositions:
            x1, y1 = facePosition.pop('x1'), facePosition.pop('y1')
            x2, y2 = facePosition.pop('x2'), facePosition.pop('y2')
            landmark = list(facePosition.values())
            face = aligned_face(img, landmark)
            
            Outputfile = output_path + file_name
            Outputfile_gray = output_path_gray + file_name
            
            cv2.imwrite(Outputfile, face[:,:,::-1])
            
            imgBGR = cv2.imread(Outputfile, cv2.IMREAD_COLOR).astype('float32')
            image48 = cv2.resize(imgBGR, (48, 48), interpolation=cv2.INTER_CUBIC)
            imgGra = cv2.cvtColor(image48, cv2.COLOR_BGR2GRAY)
            
#             cv2.imwrite(Outputfile, face[:,:,::-1])
            cv2.imwrite(Outputfile_gray, imgGra)

        st.image(Outputfile)
        st.image(Outputfile_gray)
        
        return Outputfile_gray
    
    else: 
        st.write("""沒有偵測到人臉""")
        return 
