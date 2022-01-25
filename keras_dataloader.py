import os
import numpy as np
import keras
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from skimage import color
import albumentations as A
import random

#albumentations augmentation, p- probability
aug= A.Compose([
        A.VerticalFlip(p=0.5), 
        A.HorizontalFlip(p=1),             
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue(p=1),   
        A.RandomGamma(p=0.8)])

# pre-processing routines
def divby255_normalization(img):
    img=img.astype("float32")
    img/=255.0
    return img

def local_mean_normalization(im):
    #Compute the mean for data normalization
    b_ch=np.mean(im[:,:,0])
    g_ch=np.mean(im[:,:,1])
    r_ch=np.mean(im[:,:,2])  
    # Mean substraction     
    im_ = np.array(im, dtype=np.float32)                             
    im_ -= np.array((b_ch,g_ch,r_ch))
     
    #compute the standard deviation
    b_ch=np.std(im[:,:,0])
    g_ch=np.std(im[:,:,1])
    r_ch=np.std(im[:,:,2])

    if (b_ch==0) | (g_ch==0) | (r_ch==0): #exception handling to avoid divide by zero. Replace std by std=1
        b_ch=g_ch=r_ch=1

    im_ /= np.array((b_ch,g_ch,r_ch))

    return im_

def data_mean_normalization(im):                 
    im_  = im.astype("float32")
    #Individual channel-wise mean substraction
    im_ -= np.array((196.94, 147.73, 179.14))
    #Individual channel-wise standard deviation division
    im_ /= np.array((36.35, 46.99, 35.35))        
    
    return im_   

def sampleMeanStdExcludeWhite(img):
    
    
    img  = img.astype("float32")
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    x, y = np.where((imgGray >0) & (imgGray <220))
    
    b_ch_mean=np.mean(img[x,y,0])
    g_ch_mean=np.mean(img[x,y,1])
    r_ch_mean=np.mean(img[x,y,2])
    
    b_ch_std=np.std(img[x,y,0])
    g_ch_std=np.std(img[x,y,1])
    r_ch_std=np.std(img[x,y,2])
    
    if (b_ch_std==0) | (g_ch_std==0) | (r_ch_std==0): #std=1
        img[:, :, 0] = (img[:, :, 0] - b_ch_mean)
        img[:, :, 1] = (img[:, :, 1] - g_ch_mean)
        img[:, :, 2] = (img[:, :, 2] - r_ch_mean)
    else: 
        img[:, :, 0] = (img[:, :, 0] - b_ch_mean)/b_ch_std
        img[:, :, 1] = (img[:, :, 1] - g_ch_mean)/g_ch_std
        img[:, :, 2] = (img[:, :, 2] - r_ch_mean)/r_ch_std
    
    return img

def global_imnet_mean_normalization(im):
    # Global_Mean computation of each channel and subtract from it
    b_ch=104.00699
    g_ch=116.66877
    r_ch=122.67892
                                       
    im_  = im.astype("float32")             
    #Individual channel-wise mean substraction
              
    im_ -= np.array((b_ch,g_ch,r_ch)) 
    
    return im_          

class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root, split, batch_size=32, dim=(32,32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')

        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size, *self.dim), dtype=int)
        y = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_classes), dtype="float32")

        # Generate data
        for i, image_id in enumerate(list_IDs_temp):

            image_path = os.path.join(self.image_dir, image_id + '.png')
            label_path = os.path.join(self.label_dir, image_id + '.png')

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)  

            image, label = get_random_crop(image, label, self.dim[0], self.dim[1])

            # apply augmentations
            augmented = aug(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
           
           # pre-processing
            im_ = divby255_normalization(image)
            # im_ = local_mean_normalization(image)

            # im_ = np.rollaxis((im_),2)

            X[i,] = im_

            # Store class
            label = binarylabel(label, self.n_classes)
            #label = np.reshape(label, (self.dim[0]*self.dim[1], self.n_classes))
            label = np.reshape(label, (self.dim[0],self.dim[1], self.n_classes))
            y[i,] = label
        return X, y

def binarylabel(im_label,classes):
    
    
    im_dims = im_label.shape
    
    lab=np.zeros([im_dims[0],im_dims[1],classes],dtype="float32")
    for class_index in range(classes):
        
        lab[im_label==class_index, class_index] = 1
        
    return lab

class ValDataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root, split, batch_size=32, dim=(32,32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')

        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size, *self.dim), dtype=int)
        y = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_classes), dtype="float32")

        # Generate data
        for i, image_id in enumerate(list_IDs_temp):

            image_path = os.path.join(self.image_dir, image_id + '.png')
            label_path = os.path.join(self.label_dir, image_id + '.png')
            image = cv2.imread(image_path)
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)  
            
            # random crop
            image, label = get_random_crop(image, label, self.dim[0], self.dim[1])

            # pre-processing
            im_ = divby255_normalization(image)  
            # im_ = local_mean_normalization(image)

            # im_ = np.rollaxis((im_),2)
            
            X[i,] = im_

            # Store class
            label = binarylabel(label, self.n_classes)
            #label = np.reshape(label, (self.dim[0]*self.dim[1], self.n_classes))
            label = np.reshape(label, (self.dim[0],self.dim[1], self.n_classes))
            y[i,] = label
#             print(y.shape)
        #print('val ----->', X.shape, y.shape)
        return X, y
    
def get_random_crop(image, label, crop_height, crop_width):
    '''
    get_random_crop(image, label, self.dim[0], self.dim[1])
    example_image = np.random.randint(0, 256, (1024, 1024, 3))
    random_crop = get_random_crop(example_image, 64, 64)
    '''
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    if (max_x !=0 and max_y!=0):
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
    else:
        x=0
        y=0

    im_crop = image[y: y + crop_height, x: x + crop_width,:]
    im_label = label[y: y + crop_height, x: x + crop_width]

    return im_crop,im_label
    