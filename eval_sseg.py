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


def compute_miou(gt_masks, pred_masks, num_classes=150):
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
    small_obj_labels = {18, 37, 67, 99, 109, 126, 133, 136, 148, 149}
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


saved_folder = 'output/comparison_results_ADE20K'
data_folder = '/home/yimeng/ARGO_datasets/Datasets/ADE20K/Semantic_Segmentation'
sam_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/ade20k_sam_results'
sam_vote_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/ade20k_vote_sam_with_maskFormer_results'
maskFormer_results_folder = '/home/yimeng/ARGO_scratch/sseg/MaskFormer/output/ade20k_maskformer_results'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)
# img_list = [img_list[143]]


mIoU_maskFormer_list, mIoU_vote_list = [], []
mIoU_small_objs_maskFormer_list, mIoU_small_objs_vote_list = [], []
boundary_mIoU_maskFormer_list, boundary_mIoU_vote_list = [], []

for idx in range(len(img_list)):
    img_dir = img_list[idx]['img']
    anno_dir = img_list[idx]['anno']
    name = img_dir[18:-4]
    print(f'name = {name}')
    # load gt anno
    sseg_gt = cv2.imread(f'{data_folder}/{anno_dir}', cv2.IMREAD_GRAYSCALE).astype(np.int64)

    # load maskFormer results
    sseg_maskFormer = np.load(f'{maskFormer_results_folder}/{name}.npy', allow_pickle=True)
    sseg_maskFormer += 1
    vis_maskFormer = cv2.imread(f'{maskFormer_results_folder}/{name}_mask.jpg')
    vis_maskFormer = cv2.cvtColor(vis_maskFormer, cv2.COLOR_BGR2RGB)

    # load vote results
    sseg_vote = np.load(f'{sam_vote_results_folder}/{name}.npy', allow_pickle=True)
    sseg_vote += 1

    # compute maskFormer miou
    miou_maskFormer = compute_miou(sseg_gt, sseg_maskFormer)
    print(f'miou maskFormer = {miou_maskFormer}')
    mIoU_maskFormer_list.append(miou_maskFormer)

    miou_small_obj_maskFormer = compute_miou_small_objs(sseg_gt, sseg_maskFormer)
    print(f'miou small ojbs maskFormer = {miou_small_obj_maskFormer}')
    if miou_small_obj_maskFormer:
        mIoU_small_objs_maskFormer_list.append(miou_small_obj_maskFormer)

    boundary_miou_maskFormer = compute_boundary_miou(sseg_gt, sseg_maskFormer)
    print(f'boundary miou maskFormer = {boundary_miou_maskFormer}')
    boundary_mIoU_maskFormer_list.append(boundary_miou_maskFormer)

    # compute SAM vote miou
    miou_vote = compute_miou(sseg_gt, sseg_vote)
    print(f'miou SAM vote = {miou_vote}')
    mIoU_vote_list.append(miou_vote)

    miou_small_obj_vote = compute_miou_small_objs(sseg_gt, sseg_vote)
    print(f'miou small ojbs vote = {miou_small_obj_vote}')
    if miou_small_obj_vote:
        mIoU_small_objs_vote_list.append(miou_small_obj_vote)

    boundary_miou_vote = compute_boundary_miou(sseg_gt, sseg_vote)
    print(f'boundary miou SAM vote = {boundary_miou_vote}')
    boundary_mIoU_vote_list.append(boundary_miou_vote)

    # load the image
    image = cv2.imread(f'{data_folder}/{img_dir}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # visualize vote results
    H, W = sseg_vote.shape
    vis_sseg_vote = np.ones((H, W, 3))
    unique_labels = np.unique(sseg_vote)
    for idx in unique_labels:
        vis_sseg_vote[sseg_vote == idx] = np.random.random(3)

    # load sam results
    sseg_sam = np.load(f'{sam_results_folder}/{name}.npy', allow_pickle=True)
    vis_sam = np.ones((H, W, 3))
    unique_labels = np.unique(sseg_sam)
    for idx in unique_labels:
        vis_sam[sseg_sam == idx] = np.random.random(3)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ax[0][0].imshow(image)
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].get_yaxis().set_visible(False)
    ax[0][0].set_title("Input Image")
    ax[0][0].title.set_fontsize(18)
    ax[0][1].imshow(vis_maskFormer)
    ax[0][1].get_xaxis().set_visible(False)
    ax[0][1].get_yaxis().set_visible(False)
    if miou_small_obj_maskFormer:
        ax[0][1].set_title(f"maskFormer mIoU: {miou_maskFormer:.3f}, {miou_small_obj_maskFormer:.3f} (small objs)")
    else:
        ax[0][1].set_title(f"maskFormer mIoU: {miou_maskFormer:.3f}")
    ax[0][1].title.set_fontsize(18)
    ax[1][0].imshow(vis_sam)
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title(f"SAM segments")
    ax[1][0].title.set_fontsize(18)
    ax[1][1].imshow(vis_sseg_vote)
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    if miou_small_obj_vote:
        ax[1][1].set_title(f"SAM vote mIoU: {miou_vote:.3f}, {miou_small_obj_vote:.3f} (small objs)")
    else:
        ax[1][1].set_title(f"SAM vote mIoU: {miou_vote:.3f}")
    ax[1][1].title.set_fontsize(18)
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'{saved_folder}/{name}.jpg')
    plt.close()

    # assert 1 == 2


print(f'mIoU from maskFormer is: {np.array(mIoU_maskFormer_list).mean()}')
print(f'mIoU from SAM vote is: {np.array(mIoU_vote_list).mean()}')
print(f'mIoU from maskFormer for small objs is: {np.array(mIoU_small_objs_maskFormer_list).mean()}')
print(f'mIoU from SAM vote for small objs is: {np.array(mIoU_small_objs_vote_list).mean()}')
print(f'boundary mIoU from maskFormer is: {np.array(boundary_mIoU_maskFormer_list).mean()}')
print(f'boundary mIoU from SAM vote is: {np.array(boundary_mIoU_vote_list).mean()}')
