import skimage.measure
import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator  # NOQA


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


data_folder = '/projects/kosecka/Datasets/ADE20K/Semantic_Segmentation'
saved_folder = 'output/ade20k_sam_results'

sam_checkpoint = "/scratch/yli44/segment-anything/model_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(model=sam,
                                           points_per_side=64,
                                           pred_iou_thresh=0.86,
                                           stability_score_thresh=0.92,
                                           crop_n_layers=1,
                                           crop_n_points_downscale_factor=2,
                                           min_mask_region_area=100,  # Requires open-cv to run post-processing
                                           )

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)
# img_list = ['000110000010101']


for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['img']
    name = img_dir[18:-4]
    print(f'name = {name}')
    image = cv2.imread(f'{data_folder}/{img_dir}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    masks = mask_generator.generate(image)

    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    ax.imshow(image)
    show_anns(masks)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(f'output/{name}.jpg')
    plt.close()
    '''

    img_mask = np.zeros((H, W), dtype=np.uint16)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # if len(sorted_masks) > 255:
    #    sorted_masks = sorted_masks[-255:]

    count_mask = 1
    for ann in sorted_masks:
        m = ann['segmentation']
        # color_mask = np.random.random(3)
        img_mask[m] = count_mask
        count_mask += 1

    assert count_mask - 1 == len(sorted_masks)

    # _, mask_mask = cv2.threshold(img_mask, 1, 255, cv2.THRESH_BINARY_INV)

    # img_mask_dst = cv2.inpaint(img_mask, mask_mask, 100, cv2.INPAINT_NS)

    instance_label, num_ins = skimage.measure.label(
        img_mask == 0, background=0, connectivity=1, return_num=True)

    for idx_ins in range(1, num_ins + 1):
        img_mask[instance_label == idx_ins] = count_mask
        count_mask += 1

    # reformat the labels
    unique_labels = np.unique(img_mask)
    # mapped_values = list(range(1, len(unique_labels) + 1))
    # dict_labels = {unique_labels[i]: mapped_values[i] for i in range(len(unique_labels))}

    vis_mask = np.ones((H, W, 3))
    # vis_mask[:, :, 3] = 0
    for idx in unique_labels:
        vis_mask[img_mask == idx] = np.random.random(3)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    ax.imshow(vis_mask)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{saved_folder}/{name}_mask.jpg')
    plt.close()

    print(f'min_mask = {img_mask.min()}')

    assert img_mask.min() > 0

    np.save(f'{saved_folder}/{name}.npy', img_mask)
