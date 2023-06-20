import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure


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


# a = np.zeros((100, 100), dtype=np.uint8)
# b = np.zeros((100, 100), dtype=np.uint8)

# a[20:40, 20:40] = 1
# b[30:40, 30:40] = 1

saved_folder = 'output/comparison_results_ADE20K'
data_folder = '/home/yimeng/ARGO_datasets/Datasets/ADE20K/Semantic_Segmentation'
sam_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/ade20k_sam_results'
sam_vote_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/ade20k_vote_sam_with_maskFormer_results'
maskFormer_results_folder = '/home/yimeng/ARGO_scratch/sseg/MaskFormer/output/ade20k_maskformer_results'

anno_dir = 'annotations/validation/ADE_val_00000085.png'
name = 'ADE_val_00000085'

gt_masks = cv2.imread(f'{data_folder}/{anno_dir}', cv2.IMREAD_GRAYSCALE).astype(np.int64)
pred_masks = np.load(f'{maskFormer_results_folder}/{name}.npy', allow_pickle=True)
pred_masks += 1

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

        if G_seg.sum() < 50:
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

biou = np.mean(iou_list)
