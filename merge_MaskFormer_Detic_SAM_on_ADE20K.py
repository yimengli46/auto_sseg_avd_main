'''
For evaluation on ADE20K dataset.
merge the outputs from Detic, MaskFormer and AVD instances.
'''
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from constants import ade20k_dict, lvis_dict, avd_dict, UNWANTED_CLASSES, ALLOWED_OBJECT_OVERLAY_PAIRS, ade20k_wanted_classes
from utils import _OFF_WHITE, draw_text, draw_binary_mask, comp_bbox_iou, comp_mask_iou
import _pickle as cPickle
import bz2

_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_LARGE_MASK_AREA_THRESH = 120000


# merge dataset dicts
dataset_dict = {}
dataset_dict[0] = 'void'
# merge with ade20k
start_label_idx = 0
for k, v in ade20k_dict.items():
    dataset_dict[k + start_label_idx] = v

customized2ade20k = {
    0:	13,
    1:	18,
    2:	23,
    3:	37,
    4:	38,
    5:	40,
    6:	48,
    7:	58,
    8:	66,
    9:	86,
    10:	90,
    11:	99,
    12:	116,
    13:	119,
    14:	120,
    15:	125,
    16:	126,
    17:	128,
    18:	133,
    19:	136,
    20:	139,
    21:	143,
    22:	146,
    23:	148,
    24:	149,
    25:	150,
}

saved_folder = 'output/ade20k_MaskFormer_Detic_SAM_results'
stage_a_result_folder = 'output/ade20k_Detic_results'
stage_b_result_folder = 'output/ade20k_sam_Detic_results'
stage_d_result_folder = 'output/ade20k_maskformer_results'
stage_e_result_folder = 'output/ade20k_sam_results'

data_folder = '/projects/kosecka/Datasets/ADE20K/Semantic_Segmentation'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['img']
    img_name = img_dir[18:-4]
    print(f'name = {img_name}')

    image = cv2.imread(f'{data_folder}/{img_dir}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # load sam results
    sseg_sam = np.load(f'{stage_e_result_folder}/{img_name}.npy', allow_pickle=True)

    # load maskFormer results
    sseg_maskFormer = np.load(f'{stage_d_result_folder}/{img_name}.npy', allow_pickle=True)

    H, W = sseg_sam.shape
    sseg_vote = np.zeros((H, W), dtype=np.uint16)

    # ==================================== merge with maskFormer results

    # go through each segment in sseg_sam
    segment_ids = np.unique(sseg_sam)
    for segment_id in segment_ids:
        mask = (sseg_sam == segment_id)
        # get the segment from maskFormer result
        segment = sseg_maskFormer[mask]
        counts = np.bincount(segment)
        most_common_idx = np.argmax(counts)
        sseg_vote[mask] = most_common_idx + 1

    # '''
    # ==================================== merge with Detic result
    # load Detic boxes
    with bz2.BZ2File(f'{stage_a_result_folder}/{img_name}.pbz2', 'rb') as fp:
        pred_dict = cPickle.load(fp)
        num_instances = pred_dict['num_instances']
        detic_pred_boxes = pred_dict['pred_boxes'].astype(np.int32)
        scores = pred_dict['scores']
        detic_pred_classes = pred_dict['pred_classes']

    if num_instances > 0:

        # load Detic-SAM masks
        with bz2.BZ2File(f'{stage_b_result_folder}/{img_name}_masks.pbz2', 'rb') as fp:
            detic_masks = cPickle.load(fp)

        # sort the masks ascendingly. So the small masks are in the front

        sorted_masks = sorted(enumerate(detic_masks), key=(lambda x: x[1].sum()), reverse=False)
        sorted_masks_idx, _ = zip(*sorted_masks)
        sorted_masks_idx = np.array(sorted_masks_idx)
        detic_pred_boxes = detic_pred_boxes[sorted_masks_idx]
        detic_pred_classes = detic_pred_classes[sorted_masks_idx]
        detic_masks = detic_masks[sorted_masks_idx]
        scores = scores[sorted_masks_idx]

        # step : remove the smaller mask with large mask iou with a larger mask.

        unwanted_mask_idx_list = []
        detic_mask_idx_arr = list(range(detic_masks.shape[0]))
        for idx_mask_a in detic_mask_idx_arr[:-1]:
            for idx_mask_b in detic_mask_idx_arr[idx_mask_a + 1:]:

                mask_a = detic_masks[idx_mask_a]
                mask_b = detic_masks[idx_mask_b]
                # if IoU > thresh, put in unwanted list
                iou = comp_mask_iou(mask_a, mask_b)
                if iou > 0.5:
                    # # step : check if the overlay between obj_a and obj_b is allowed
                    # class_a = dataset_dict[detic_pred_classes[idx_mask_a] + start_label_idx]
                    # class_b = dataset_dict[detic_pred_classes[idx_mask_b] + start_label_idx]

                    # if class_a in ALLOWED_OBJECT_OVERLAY_PAIRS and class_b in ALLOWED_OBJECT_OVERLAY_PAIRS[class_a]:
                    #     print(f'class_a = {class_a}, class_b = {class_b}')
                    #     pass
                    # else:
                    #     unwanted_mask_idx_list.append(idx_mask_a)
                    unwanted_mask_idx_list.append(idx_mask_a)

        idx_mask = np.isin(np.array(range(num_instances), dtype=int), unwanted_mask_idx_list, invert=True)
        detic_pred_boxes = detic_pred_boxes[idx_mask]
        detic_pred_classes = detic_pred_classes[idx_mask]
        detic_masks = detic_masks[idx_mask]
        scores = scores[idx_mask]
        num_instances = idx_mask.sum()

        # convert all the masks into one simgle mask
        Detic_mask = np.zeros((H, W), dtype=np.uint16)
        for idx_mask in range(detic_masks.shape[0]):
            mask = detic_masks[idx_mask]
            mask_class = customized2ade20k[detic_pred_classes[idx_mask]]
            print(f'id = {mask_class}, class = {dataset_dict[mask_class]}')
            if mask_class in [18, 37, 40, 58, 126, 136, 149]:
                continue
            Detic_mask[mask] = mask_class

        sseg_vote = np.where(Detic_mask > 0, Detic_mask, sseg_vote)
    # '''
    # ================= visualization
    # vis_sseg_vote = np.ones((H, W, 3))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    ax.imshow(image)
    unique_labels = list(np.unique(sseg_vote))
    if 0 in unique_labels:
        unique_labels = unique_labels[1:]

    for label in unique_labels:
        binary_mask = (sseg_vote == label).astype(np.uint8)
        mask_color = np.random.random(3)
        text = dataset_dict[label]
        draw_binary_mask(ax,
                         binary_mask,
                         color=mask_color,
                         edge_color=_OFF_WHITE,
                         text=text,
                         alpha=0.8,
                         )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{saved_folder}/{img_name}_sseg.jpg')
    plt.close()

    cv2.imwrite(f'{saved_folder}/{img_name}_labels.png', sseg_vote)

    np.save(f'{saved_folder}/{img_name}.npy', sseg_vote)
