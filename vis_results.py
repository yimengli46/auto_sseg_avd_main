import numpy as np
import matplotlib.pyplot as plt
import cv2
# from eval_sseg import compute_miou


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


saved_folder = 'output/comparison_results'
data_folder = '/home/yimeng/ARGO_datasets/Datasets/ADE20K/Semantic_Segmentation'
sam_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/ade20k_sam_results'
sam_vote_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/vote_sam_with_maskFormer_results'
maskFormer_results_folder = '/home/yimeng/ARGO_scratch/sseg/MaskFormer/output/ade20k_maskformer_results'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

mIoU_maskFormer_list, mIoU_vote_list = [], []
mIoU_small_objs_maskFormer_list, mIoU_small_objs_vote_list = [], []

for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['img']
    anno_dir = img_list[idx]['anno']
    name = img_dir[18:-4]

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

    # compute SAM vote miou
    miou_vote = compute_miou(sseg_gt, sseg_vote)
    print(f'miou SAM vote = {miou_vote}')
    mIoU_vote_list.append(miou_vote)

    miou_small_obj_vote = compute_miou_small_objs(sseg_gt, sseg_vote)
    print(f'miou small ojbs vote = {miou_small_obj_vote}')
    if miou_small_obj_vote:
        mIoU_small_objs_vote_list.append(miou_small_obj_vote)

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
    ax[0][0].set_title("rgb")
    ax[0][1].imshow(vis_maskFormer)
    ax[0][1].get_xaxis().set_visible(False)
    ax[0][1].get_yaxis().set_visible(False)
    if miou_small_obj_maskFormer:
        ax[0][1].set_title(f"maskFormer mIoU: {miou_maskFormer:.3f}, {miou_small_obj_maskFormer:.3f} (small objs)")
    else:
        ax[0][1].set_title(f"maskFormer mIoU: {miou_maskFormer:.3f}")
    ax[1][0].imshow(vis_sam)
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title(f"SAM")
    ax[1][1].imshow(vis_sseg_vote)
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    if miou_small_obj_vote:
        ax[1][1].set_title(f"SAM vote mIoU: {miou_vote:.3f}, {miou_small_obj_vote:.3f} (small objs)")
    else:
        ax[1][1].set_title(f"SAM vote mIoU: {miou_vote:.3f}")
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'{saved_folder}/{name}.jpg')
    plt.close()

    # assert 1 == 2


print(f'mIoU from maskFormer is: {np.array(mIoU_maskFormer_list).mean()}')
print(f'mIoU from SAM vote is: {np.array(mIoU_vote_list).mean()}')
print(f'mIoU from maskFormer for small objs is: {np.array(mIoU_small_objs_maskFormer_list).mean()}')
print(f'mIoU from SAM vote for small objs is: {np.array(mIoU_small_objs_vote_list).mean()}')
