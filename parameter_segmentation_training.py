'''
Extract road features from kaggle-massachusetts-roads-dataset using semantic segmentation
( https://www.kaggle.com/insaff/massachusetts-roads-dataset )
Chanukya Krishna Chama - 25th Jan 2022
'''
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras import layers
from keras.models import *
from keras.layers import *
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Sequential
from keras import callbacks
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from histNet.keras_dataloader import DataGenerator, ValDataGenerator
import segmentation_models as sm
sm.set_framework('tf.keras')
keras.backend.set_image_data_format('channels_last') #effnet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(1337)  # for reproducibility

def set_tf1_gpu_option(gpu,memory_fraction):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config= config))
    return

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 4e-3
    if epoch >=50 and epoch <100: 
        lr *= 0.5
    elif epoch >= 100 and epoch <150:
        lr *= 0.25
    elif epoch >= 150: 
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

BACKBONE = 'efficientnetb4'
classes=2 # current dataset num_classes
path2write = '/workspace-downloads/Training/segmentation_roads_ueffnet/v1-4e-3-50step-ueffnetb4-divby255-bcediceloss-p50/'
if not os.path.exists(path2write):
    os.makedirs(path2write)

# preTrained =''
# new_classes=2 # pretrained model num_classes
# preTrained = None
new_classes=classes

#Define the Image Size
batchsize = 8
depth=3
height=512
width=512

root_folder = '/workspace-downloads/data/road_segmentation_ideal/training/'
# Parameters
params = {'dim': (height, width),
          'batch_size': batchsize,
          'n_classes': classes,
          'n_channels': depth,
          'shuffle': True}
# Generators
training_generator = DataGenerator(root_folder, 'train',  **params)
validation_generator = ValDataGenerator(root_folder, 'val',  **params)

model = sm.Unet(BACKBONE, classes=new_classes, activation='softmax', input_shape=(height, width, depth), encoder_weights='imagenet')
# model.load_weights(preTrained)
print(model.summary())

############## using segmentation module
opt = Adam(lr=lr_schedule(0))
loss=sm.losses.bce_dice_loss
# loss=sm.losses.bce_jaccard_loss
# loss=sm.losses.dice_loss
# loss=sm.losses.categorical_crossentropy
# metrics=[sm.metrics.f1_score]
metrics=[sm.metrics.iou_score,sm.metrics.f1_score]
model.compile(opt, loss, metrics)
filepath=path2write+'ueffnetb4-{epoch:02d}-{val_loss:02f}-{val_iou_score:.2f}-{val_f1-score:.2f}.h5'
modelCheck = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only= False, mode='auto', period=10)
lr_scheduler = LearningRateScheduler(lr_schedule)

class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

tensorboard_callback = LRTensorBoard(path2write)
early_stopping_callback = EarlyStopping(monitor='val_iou_score', patience=50, mode='max', verbose=1)

unn_2= model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=300, verbose=2, use_multiprocessing=True,
                    workers=6, callbacks=[tensorboard_callback,modelCheck,lr_scheduler,early_stopping_callback])


