'''
Testing
'''
 
import glob
import os
import numpy as np
import cv2
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(0)

import segmentation_models as sm

#===========================================================#
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
np.random.seed(1337)  # for reproducibility
import tensorflow as tf

def set_tf1_gpu_option(gpu,memory_fraction):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config= config))
    return
#=============================================================#
def divby255_normalization(img):
    img=img.astype("float32")
    img/=255.0
    return img
# ===========================================================================
def testing_module(Image, model, classes):   

    im_ = divby255_normalization(Image)        # pre-processing

    data = []
    data.append(((im_))) 
#       
    temp = np.array(data)   
    prob = model.predict(temp.reshape(1,height,width,3))
    prediction = np.argmax(prob[0],axis=-1)
    prediction = np.expand_dims(prediction, axis = -1)

    scale = np.uint8(255/(classes-1))
    norm_image = scale*np.uint8(prediction) 
    
    return  norm_image  
#==================================================================#
K.clear_session()

classes=2
depth=3
height=1536 
width=1536 
decoderPath= '/workspace-downloads/Training/segmentation_roads_ueffnet/v1-4e-3-50step-ueffnetb4-divby255-bcediceloss-p50/ueffnetb4-170-0.184014-0.80-0.88.h5'
BACKBONE = 'efficientnetb4'
model = sm.Unet(BACKBONE, classes=classes, activation='softmax', input_shape=(height, width, depth),encoder_weights=None)

# model.summary() 
print ("Model load :") 

input_shape=(height, width, depth)

data_path='/workspace-downloads/data/road_segmentation_ideal/testing/all-output/'
# path2write = '/workspace-downloads/data/road_segmentation_ideal/training/train/all-output/'
path2write=data_path

if not os.path.exists(path2write):
    os.makedirs(path2write)
    
model.load_weights(decoderPath) 

data_extention='.png'

paths = sorted(glob.glob(data_path + '*'+data_extention))  

all_files = [f for f in os.listdir(data_path)]
 
for filename in range(len(paths)):
    tdp = paths[filename]            
    OIm = image = cv2.imread(tdp)
#=================== Resize=============================================#        
    Rw, Cl, Cr = image.shape
    dim = (height,width)
    OIm = cv2.resize(OIm, dim, interpolation=cv2.INTER_NEAREST) #resize
#=======================================================================#  
    out_img = testing_module(OIm, model, classes)
    tilename = tdp.split('/')
    tileNum = tilename[-1].split(data_extention)    
#=================== Resize=============================================#        
    out_img = cv2.resize(out_img, (Cl, Rw), interpolation=cv2.INTER_NEAREST) #resize
#=======================================================================#  
    cv2.imwrite(path2write  + tileNum[0] + '_modlabel_exp1-170_ueb4_divby255.png', out_img)  
     
print("END")
