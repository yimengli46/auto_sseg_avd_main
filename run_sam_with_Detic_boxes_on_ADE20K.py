'''
This script is for evaluation on ADE20K.
run SAM on AVD with Detic detected bbox as prompt.
'''
import _pickle as cPickle
import bz2
import skimage.measure
import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator  # NOQA


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def segment(sam_predictor, image, xyxy):
    '''
    single box segmentation
    '''
    sam_predictor.set_image(image)
    masks = []
    for box in pred_boxes:
        mask, _, _ = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        masks.append(mask)
    masks = np.array(masks)
    return np.array(masks)


def batch_segment_input_points_and_boxes(sam_predictor, image, boxes, masks):
    H, W = image.shape[:2]
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    image_coords = np.stack((xv, yv), axis=2)

    sam_predictor.set_image(image)
    boxes = torch.tensor(pred_boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, (H, W))

    # prepare points prompt
    B = boxes.shape[0]
    input_points = np.zeros((B, 2))
    for i in range(B):
        mask = masks[i]
        mask_coords = image_coords[mask]
        center_point = np.mean(mask_coords, axis=0)
        norm = np.linalg.norm(mask_coords - center_point[None, :], axis=1)
        min_idx = np.argmin(norm)
        min_point = mask_coords[min_idx, :]
        input_points[i] = min_point

    input_points = torch.tensor(input_points, device=sam_predictor.device).unsqueeze(1)
    transformed_points = sam_predictor.transform.apply_coords_torch(input_points, image.shape[:2])

    input_labels = torch.ones((B, 1), device=sam_predictor.device)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=transformed_points,
        point_labels=input_labels,
        boxes=transformed_boxes,
        multimask_output=False
    )

    masks = masks.cpu().numpy()

    return masks


def batch_segment_input_boxes(sam_predictor, image, boxes):
    sam_predictor.set_image(image)
    boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, (H, W))

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    masks = masks.cpu().numpy()

    return masks


# =========================== initialize sam model =================================
sam_checkpoint = "../segment-anything/model_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)


# ============================= run on AVD =======================================

data_folder = '/projects/kosecka/Datasets/ADE20K/Semantic_Segmentation'
saved_folder = 'output/ade20k_sam_Detic_results'
stage_a_result_folder = 'output/ade20k_Detic_results'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['img']
    img_name = img_dir[18:-4]
    print(f'name = {img_name}')

    image = cv2.imread(f'{data_folder}/{img_dir}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # load Detic boxes
    with bz2.BZ2File(f'{stage_a_result_folder}/{img_name}.pbz2', 'rb') as fp:
        pred_dict = cPickle.load(fp)
        num_instances = pred_dict['num_instances']
        pred_boxes = pred_dict['pred_boxes'].astype(np.int32)
        scores = pred_dict['scores']
        pred_classes = pred_dict['pred_classes']
        # pred_masks = pred_dict['pred_masks']

    if num_instances == 0:
        continue
    # run SAM
    masks = batch_segment_input_boxes(sam_predictor, image, pred_boxes)
    masks = masks[:, 0]  # num_masks x h x w

    # sort the masks decendingly. So the large masks in the front
    sorted_masks = sorted(enumerate(masks), key=(lambda x: x[1].sum()), reverse=True)
    sorted_bbox_idx, _ = zip(*sorted_masks)

    # convert all the masks into one simgle mask
    img_mask = np.zeros((H, W), dtype=np.uint16)
    for idx_bbox in list(sorted_bbox_idx):
        mask = masks[idx_bbox]
        mask_class = pred_classes[idx_bbox]
        img_mask[mask] = mask_class

    # for visualize the built mask
    unique_labels = np.unique(img_mask)
    vis_mask = np.zeros((H, W, 3))
    # skip label 0
    for idx in unique_labels[1:]:
        vis_mask[img_mask == idx] = np.random.random(3)

    # visualization bbox and mask
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 30))
    ax[0].imshow(image)
    for idx in range(masks.shape[0]):
        # print(f'mask.shape = {mask.shape}')
        show_mask(masks[idx], ax[0], random_color=True)
    for box in pred_boxes:
        show_box(box, ax[0])
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].imshow(vis_mask)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{saved_folder}/{img_name}_vis.jpg')
    plt.close()

    cv2.imwrite(f'{saved_folder}/{img_name}_mask_labels.png', img_mask)

    with bz2.BZ2File(f'{saved_folder}/{img_name}_masks.pbz2', 'w') as fp:
        cPickle.dump(
            masks,
            fp
        )
