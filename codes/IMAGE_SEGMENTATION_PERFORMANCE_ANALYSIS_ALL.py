'''
Performance analysis
'''
'''
This script computes various metrics for image segmentation tasks
Segmentation output and labels must have same names in different directory
Segmentation output and labels must have same dimensions and class values
This scrip works with multiclass outputs as well
Background must have class label '0', Non-zero class labels are considered as foreground objects
All the class labels must be 0, 1, 2,.. and so on

'''
 
import numpy as np
import cv2
import sklearn.metrics as metrics
from pathlib import Path

# Macro Defination
LABEL_PATH = '/workspace-downloads/data/road_segmentation_ideal/testing/output_cleaned/' 
# OUTPUT_PATH=LABEL_PATH
OUTPUT_PATH = '/workspace-downloads/data/road_segmentation_ideal/testing/all-output/'
NO_OF_CLASSES = 2
 
LABEL_SUFFIX = '.png'
OUTPUT_SUFFIX = '_modlabel_exp1-170_ueb4_divby255.png'
  
confusion_matrix = np.zeros([NO_OF_CLASSES, NO_OF_CLASSES])
 
output_file_name_list = Path(OUTPUT_PATH).glob('*' + OUTPUT_SUFFIX)
for output_file_name in output_file_name_list:
    print(output_file_name)
    
    ground_truth_name = str(Path(LABEL_PATH) / (str(output_file_name.name).split(OUTPUT_SUFFIX)[0] + LABEL_SUFFIX))
    output_image = cv2.imread(str(output_file_name), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(ground_truth_name, cv2.IMREAD_GRAYSCALE)
    print(np.unique(output_image))
    print(np.unique(ground_truth))
 
    # Uncomment below code to bring the labels in sequence of [0, 1, 2,... , NO_OF_CLASSES - 1]
    output_image = (np.around((NO_OF_CLASSES - 1) * output_image.astype(np.float32) / 255)).astype(np.uint8)
    # ground_truth = (np.around((NO_OF_CLASSES - 1) * ground_truth.astype(np.float32) / 255)).astype(np.uint8)
 
    confusion_matrix+= metrics.confusion_matrix(ground_truth.reshape(-1), output_image.reshape(-1),labels=[0,1])
    # confusion_matrix+= metrics.confusion_matrix(ground_truth.reshape(-1), output_image.reshape(-1), range(NO_OF_CLASSES))

    print(confusion_matrix)
 
 
total_predictions = np.sum(confusion_matrix)
mean_accuracy = mean_iou = mean_dice = 0
for class_id in range(0, NO_OF_CLASSES):
    # tn, fp, fn, tp = confusion_matrix.ravel()
    tp = confusion_matrix[class_id, class_id]
    fp = np.sum(confusion_matrix[: class_id, class_id]) + np.sum(confusion_matrix[class_id + 1 :, class_id])
    fn = np.sum(confusion_matrix[class_id, : class_id]) + np.sum(confusion_matrix[class_id, class_id + 1 :])
    tn = total_predictions - tp - fp - fn
    
    accuracy = (tp + tn) / (tn + fn + tp + fp) 
    mean_accuracy += accuracy
 
    if ((tp + fp + fn) != 0):
        iou = (tp) / (tp + fp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)
    else:
        # When there are no positive samples and model is not having any false positive, we can not judge IOU or Dice score
        # In this senario we assume worst case IOU or Dice score. This also avoids 0/0 condition
        iou = 0.0
        dice = 0.0
 
    mean_iou += iou
    mean_dice += dice
 
    print("CLASS: {}: Accuracy: {}, IOU: {}, Dice: {}".format(class_id, accuracy, iou, dice))
 
mean_accuracy = mean_accuracy / (NO_OF_CLASSES)
mean_iou = mean_iou / (NO_OF_CLASSES)
mean_dice = mean_dice / (NO_OF_CLASSES)
print("Mean Accuracy: {}, Mean IOU: {}, Mean Dice: {}".format(mean_accuracy, mean_iou, mean_dice))