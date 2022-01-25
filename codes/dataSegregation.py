'''
Data Segregation to train & val splits
'''
import os
import numpy as np
import cv2
import random
from shutil import copyfile

path = '/workspace-downloads/data/road_segmentation_ideal/training/'
data_path = path+'input/'
clean_label_path = path+'output_cleaned/'

os.makedirs(os.path.join(path, 'train/train_data_all/'))
os.makedirs(os.path.join(path, 'train/train_labels_all/'))
os.makedirs(os.path.join(path, 'train/train_segmentation/'))
os.makedirs(os.path.join(path, 'val/val_data_all/'))
os.makedirs(os.path.join(path, 'val/val_labels_all/'))
os.makedirs(os.path.join(path, 'val/val_segmentation/'))

all_files = [f for f in os.listdir(clean_label_path)]
num_images = len(all_files)
random.shuffle(all_files)
num_images_train = int(round(num_images*0.7)) #70%
num_images_val = num_images - num_images_train
train_list = all_files[:num_images_train]
val_list = all_files[num_images_train:]
random.shuffle(all_files)
count = 0
for filename in train_list:
    fname = filename[0:-4]+'.png'
    copyfile(os.path.join(data_path, fname), os.path.join(path, 'train/train_data_all/', fname))
    copyfile(os.path.join(clean_label_path, filename), os.path.join(path, 'train/train_labels_all/', filename))

for filename in val_list:
    fname = filename[0:-4]+'.png'
    copyfile(os.path.join(data_path, fname), os.path.join(path, 'val/val_data_all/', fname))
    copyfile(os.path.join(clean_label_path, filename), os.path.join(path, 'val/val_labels_all/', filename))

with open(path+'train/train_segmentation/'+'train.txt', 'w') as f:
    for item in train_list:
        f.write("%s\n" % item[:-4])

with open(path+'val/val_segmentation/'+'val.txt', 'w') as f2:
    for item in val_list:
        f2.write("%s\n" % item[:-4])
