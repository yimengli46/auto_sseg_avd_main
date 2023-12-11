'''
This script computes the mean intersection over union (mIoU) and mIoU for small objects
for the MaskFormer semantic segmentation results and SAM voting semantic segmentation results
on the ADE20K dataset. 

It also visualizes the results for 4 images, including the original image,
MaskFormer result, SAM segmentation result, and SAM voting results. 

The computed mIoUs are written on the visualizations.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.measure


def compute_iou(gt_mask, pred_mask, class_label=1):
    # Compute intersection
    intersection = np.logical_and(gt_mask == class_label, pred_mask == class_label).sum()
    # print(f'intersection = {intersection}')
    # Compute union
    union = np.logical_or(gt_mask == class_label, pred_mask == class_label).sum()

    # Compute IoU
    iou = intersection / (union + 1e-10)  # Add small epsilon to avoid division by zero
    # print(f'iou = {iou}')
    return iou


def compute_miou(gt_masks, pred_masks, num_classes=1600):
    miou = 0.0
    gt_class_labels = set(np.unique(gt_masks))
    pred_class_labels = set(np.unique(pred_masks))
    class_labels = gt_class_labels.union(pred_class_labels)
    for class_label in class_labels:
        if class_label == 0:
            continue
        iou = compute_iou(gt_masks, pred_masks, class_label)
        miou += iou

    miou /= len(class_labels) - 1

    return miou


def compute_miou_small_objs(gt_masks, pred_masks):
    miou = 0.0
    gt_class_labels = set(np.unique(gt_masks))
    small_obj_labels = {986, 968, 289, 765, 494, 283, 1305}
    class_labels = gt_class_labels.intersection(small_obj_labels)
    for class_label in class_labels:
        if class_label == 0:
            continue
        iou = compute_iou(gt_masks, pred_masks, class_label)
        miou += iou

    if len(class_labels) > 0:
        miou /= len(class_labels)
    else:
        miou = None

    return miou


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def compute_boundary_miou(gt_masks, pred_masks, small_segment_num_pixel_thresh=50):
    # Initialize IoU_list
    iou_list = []

    gt_class_labels = set(np.unique(gt_masks))
    # For each category,
    for class_id in gt_class_labels:
        # Find all the segments S_G belong to this category in G
        gt_current_class = (gt_masks == class_id)
        S_G, num_G_seg = skimage.measure.label(gt_current_class, background=0, connectivity=2, return_num=True)

        # Find all the segments S_P belong to this category in P
        pred_current_class = (pred_masks == class_id)
        S_P, num_P_seg = skimage.measure.label(pred_current_class, background=0, connectivity=2, return_num=True)

        # For each segment in S_G,
        for idx_G_seg in range(1, num_G_seg + 1):
            G_seg = (S_G == idx_G_seg)

            if G_seg.sum() < small_segment_num_pixel_thresh:
                continue

            if num_P_seg > 0:
                max_iou_P_seg_id = 1
                max_iou = -1
                for idx_P_seg in range(1, num_P_seg):
                    P_seg = (S_P == idx_P_seg)
                    iou = compute_iou(G_seg, P_seg)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_P_seg_id = idx_P_seg

                max_iou_P_seg = (S_P == max_iou_P_seg_id).astype(np.uint8)
                G_seg = G_seg.astype(np.uint8)

                b_iou = boundary_iou(G_seg, max_iou_P_seg)
                iou_list.append(b_iou)
            else:
                iou_list.append(0)

    boundary_miou = np.mean(iou_list)
    return boundary_miou


gt_folder = 'multi_view_verification_handpicked_frames/gt'

single_view_results_folder = 'multi_view_verification_handpicked_frames/single'
multi_view_results_folder = 'multi_view_verification_handpicked_frames/figure'


mIoU_single_list, mIoU_multi_list = [], []
mIoU_small_objs_single_list, mIoU_small_objs_multi_list = [], []
# boundary_mIoU_maskFormer_list, boundary_mIoU_vote_list = [], []

list_idx = [1, 2, 3, 4, 5, 7, 8, 9]
for idx in list_idx:

    # load gt anno
    sseg_gt = cv2.imread(f'{gt_folder}/0001100000{idx}0101_gt.png', cv2.IMREAD_UNCHANGED).astype(np.int64)
    sseg_multi = cv2.imread(f'{multi_view_results_folder}/0001100000{idx}01_prop_labels.png',
                            cv2.IMREAD_UNCHANGED).astype(np.int64)

    sseg_single = cv2.imread(f'{single_view_results_folder}/0001100000{idx}0101_labels.png',
                             cv2.IMREAD_UNCHANGED).astype(np.int64)

    # compute single view miou
    miou_single = compute_miou(sseg_gt, sseg_single)
    print(f'miou single = {miou_single}')
    mIoU_single_list.append(miou_single)

    miou_small_obj_single = compute_miou_small_objs(sseg_gt, sseg_single)
    print(f'miou small ojbs single = {miou_small_obj_single}')
    if miou_small_obj_single:
        mIoU_small_objs_single_list.append(miou_small_obj_single)

    # compute multiview miou
    miou_multi = compute_miou(sseg_gt, sseg_multi)
    print(f'miou multi = {miou_multi}')
    mIoU_multi_list.append(miou_multi)

    miou_small_obj_multi = compute_miou_small_objs(sseg_gt, sseg_multi)
    print(f'miou small ojbs multi = {miou_small_obj_multi}')
    if miou_small_obj_multi:
        mIoU_small_objs_multi_list.append(miou_small_obj_multi)

    print('------------------------------------------------------------------')

print(f'mIoU from singleview is: {np.array(mIoU_single_list).mean()}')
print(f'mIoU small obj from singleview is: {np.array(mIoU_small_objs_single_list).mean()}')
print(f'mIoU from multiview is: {np.array(mIoU_multi_list).mean()}')
print(f'mIoU small obj from multiview is: {np.array(mIoU_small_objs_multi_list).mean()}')
