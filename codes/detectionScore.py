'''
Detection scores
'''

import cv2
import numpy as np
import os
import imgprocRoutines

op_path = "/workspace-downloads/data/road_segmentation_ideal/training/train/all-output/"
gt_path = "/workspace-downloads/data/road_segmentation_ideal/training/train/train_labels_all/"

op_ext = '.png'
gt_ext = '.png'

op_premble = '_modlabel_exp1-170_ueb4_divby255'
gt_premble = ''

op_label = 255
gt_label = 1

#get all gt names
all_gt_names = [f for f in os.listdir(gt_path) if f.endswith(gt_premble + gt_ext)] 
# print(all_gt_names)

gt_blobs = 0
op_blobs = 0
tp_blobs = 0
fp_blobs = 0
fn_blobs = 0
tp_overlap_th = 0.1 #true positive overlap threshold, higher than this
fp_overlap_th = 0.1 #false positive overlap threshold, lower than this
tn_overlap_th = 0.1 # true negative overlap threshold, lower than this

#if you want to remove small areas from output
remove_small_blobs = True
small_blob_th = 5 #pixels #5-default

for file_idx, file_name in enumerate(all_gt_names):
    
    #print(os.path.join( gt_path, file_name.split('.')[0] +  gt_premble + gt_ext))
    #print(os.path.join( op_path, file_name.split('.')[0] +  op_premble + op_ext))
    img_gt = cv2.imread( os.path.join( gt_path, file_name[:-4]  + gt_premble + gt_ext) )
    img_op = cv2.imread( os.path.join( op_path, file_name[:-4] + op_premble + op_ext) )
    print(np.unique(img_op))
    if img_gt is None:
        print( "Not Found : ", os.path.join( gt_path, file_name ) )
        continue
    if img_op is None:
        print( "Not Found : ", os.path.join( op_path, file_name ) )
        continue
    
    img_gt = 255*np.uint8(img_gt[:, :, 0] == gt_label)
    img_op = 255*np.uint8(img_op[:, :, 0] == op_label) 
    
    img_gt = cv2.morphologyEx(img_gt, cv2.MORPH_CLOSE, imgprocRoutines.strel(size = 5).disk())
    img_op = cv2.morphologyEx(img_op, cv2.MORPH_CLOSE, imgprocRoutines.strel(size = 5).disk())
    
    
    if remove_small_blobs:
        img_op = imgprocRoutines.bwareaopen(img_op, small_blob_th).remove_small()
        img_gt = imgprocRoutines.bwareaopen(img_gt, small_blob_th).remove_small()
    
    #get connected components
    cc_gt = cv2.connectedComponents(img_gt)
    cc_op = cv2.connectedComponents(img_op)
    
    gt_blobs = gt_blobs + cc_gt[0] - 1 #always 1 blob for bg
    op_blobs = op_blobs + cc_op[0] - 1 #always 1 blob for bg
    
    #first, check for true positives / how many gt blobs correctly detected
    for c in range(1, cc_gt[0]):
        #get gt component
        comp_gt = cc_gt[1] == c
        #overlap with op blob
        overlap = np.count_nonzero((comp_gt > 0) & (cc_op[1] > 0))/np.count_nonzero(comp_gt)
        
        if overlap >= tp_overlap_th:
            tp_blobs = tp_blobs + 1
        if overlap < tn_overlap_th:
            fn_blobs = fn_blobs + 1
            
    #now check for false positives
    for c in range(1, cc_op[0]):
        #get op component
        comp_op = cc_op[1] == c
        #overlap with gt blob
        overlap = np.count_nonzero((comp_op > 0) & (cc_gt[1] > 0))/np.count_nonzero(comp_op)
        
        if overlap < fp_overlap_th:
            fp_blobs = fp_blobs + 1
    
    print('Done : ', os.path.join( gt_path, file_name ))
    print('gt_blobs = {0}, op_blobs = {1}, tp_blobs = {2}, fp_blobs = {3}, fn_blobs = {4}'
      .format(gt_blobs, op_blobs, tp_blobs, fp_blobs, fn_blobs))
    
print('gt_blobs = {0}, op_blobs = {1}, tp_blobs = {2}, fp_blobs = {3}, fn_blobs = {4}'
      .format(gt_blobs, op_blobs, tp_blobs, fp_blobs, fn_blobs))
precision = (tp_blobs / (tp_blobs + fp_blobs))
recall = (tp_blobs / (tp_blobs + fn_blobs))
fscore = 2*(precision*recall)/(precision+recall)

print('precision - ', precision)
print('recall - ', recall)
print('f-score - ', fscore)


#Prn = ALLTP/(ALLTP+ALLFP)
#Recall
# Rcl = ALLTP/(ALLTP+ALLFN)
#Fscore
# FScore = 2*(Prn*Rcl)/(Prn+Rcl)

# print(Prn, Rcl, FScore)