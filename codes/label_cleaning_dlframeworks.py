'''
Label cleaning
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
label_path = '/workspace-downloads/data/road_segmentation_ideal/testing/output/'
clean_label_path='/workspace-downloads/data/road_segmentation_ideal/testing/output_cleaned/'

if not os.path.exists(clean_label_path):
    os.makedirs(clean_label_path)

all_files = [f for f in os.listdir(label_path)]

for fileName in all_files:

    im_bgr = cv2.imread(os.path.join(label_path, fileName))
    print(fileName)
    print(im_bgr.shape)

    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

    print(np.unique(im_gray))
    rows, cols = im_gray.shape
    imL = np.zeros([rows, cols], dtype="uint8")

    imL[np.where(im_gray == 150)] = 0 # lightgreen deeplense

    imL[np.where(im_gray == 92)] = 0  # red
    imL[np.where(im_gray == 195)] = 0 # yellow
    imL[np.where(im_gray == 135)] = 0 # gray
    imL[np.where(im_gray == 154)] = 0 # green
    imL[np.where(im_gray == 80)] = 0 # blue
    imL[np.where(im_gray == 91)] = 0 # purple
    imL[np.where(im_gray == 159)] = 0 # light pink
    imL[np.where(im_gray == 146)] = 0 # brown
    imL[np.where(im_gray == 255)] = 1 # white
    imL[np.where(im_gray == 151)] = 0 # orange
    
    # opFilename = fileName[0:-13]
    opFilename = fileName[0:-4]
    
    opFilename=opFilename+'.png'
#     im_color = cv2.applyColorMap(imL, cv2.COLORMAP_JET)
#     cv2.imwrite(os.path.join(clean_label_path, opFilename ), im_color)    
    cv2.imwrite(os.path.join(clean_label_path, opFilename ), imL)
    
