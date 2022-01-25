Road segmentation from satellite imagery
############################################################################################
Machine configuration used
Ubuntu==20.04 , 24GB RTX GPU, CUDA==11.4 and CUDNN=8.0
############################################################################################
The following requirements need to be installed before starting the experiment
tensorflow=2.6.0
tensorflow-gpu=2.6.2
keras=2.6.0
numpy==1.19.5
scikit-image=0.17.2
scikit-learn==0.24.2
scipy==1.5.4
matplotlib==3.3.4
opencv-python==4.4.0.44 
openslide-python==1.1.1
python==3.6
albumentations==1.1.0
segmentation-models==1.0.1
efficientnet==1.1.1
############################################################################################

Files present in the root directory ./ are
parameter_segmentation_training.py -- main training script
Testing_script_efficientnet.py -- testing script

./histNet/keras_dataloader.py -- data loader containing augmentation, data generator

Files present in the ./codes/ directory are 
label_cleaning_dlframeworks -- used to convert label images to 0,1,2 .... etc
dataSegregation.py -- used to split the dataset to train and val for model training
IMAGE_SEGMENTATION_PERFORMANCE_ANALYSIS_ALL -- used to calculate class-wise accuracy,iou,dice
detectionScore.py -- used to calculate object detection scores precision, recall
imgprocRoutines.py -- contains list of general morphological routines
