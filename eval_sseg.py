import numpy as np
import matplotlib.pyplot as plt
import cv2


def compute_iou(gt_mask, pred_mask, class_label):
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


data_folder = '/home/yimeng/ARGO_datasets/Datasets/ADE20K/Semantic_Segmentation'
sam_vote_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/vote_sam_with_maskFormer_results'
maskFormer_results_folder = '/home/yimeng/ARGO_scratch/sseg/MaskFormer/output/ade20k_maskformer_results'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

mIoU_maskFormer_list, mIoU_vote_list = [], []
for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['anno']
    name = img_dir[23:-4]

    # load gt anno
    sseg_gt = cv2.imread(f'{data_folder}/{img_dir}', cv2.IMREAD_GRAYSCALE).astype(np.int64)

    # load maskFormer results
    sseg_maskFormer = np.load(f'{maskFormer_results_folder}/{name}.npy', allow_pickle=True)
    sseg_maskFormer += 1

    # load vote results
    sseg_vote = np.load(f'{sam_vote_results_folder}/{name}.npy', allow_pickle=True)
    sseg_vote += 1

    miou = compute_miou(sseg_gt, sseg_maskFormer)
    print(f'miou = {miou}')
    mIoU_maskFormer_list.append(miou)

    miou = compute_miou(sseg_gt, sseg_vote)
    print(f'miou = {miou}')
    mIoU_vote_list.append(miou)

print(f'mIoU from maskFormer is: {np.array(mIoU_maskFormer_list).mean()}')
print(f'mIoU from SAM vote is: {np.array(mIoU_vote_list).mean()}')
