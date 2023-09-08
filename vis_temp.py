"""
This script is for visualization when writing the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from constants import ade20k_dict, lvis_dict, avd_dict, UNWANTED_CLASSES, ALLOWED_OBJECT_OVERLAY_PAIRS, ade20k_wanted_classes
from utils import _OFF_WHITE, draw_binary_mask, comp_bbox_iou, comp_mask_iou
from constants import colormap

folder = 'temp/temp_maskformer'
img_name = '000810000120101'
data_folder = 'data/ActiveVisionDataset'
label_folder = 'output/stage_d_maskFormer_results'
scene = 'Home_008_1'

COLOR = colormap(rgb=True)

dataset_dict = {}
dataset_dict[0] = 'void'
# merge with ade20k
start_label_idx = 0
for k, v in ade20k_dict.items():
    dataset_dict[k + start_label_idx] = v
# merge with lvis
start_label_idx = 150
for k, v in lvis_dict.items():
    dataset_dict[k + start_label_idx] = v
# merge with avd instance
start_label_idx = 1500
for k, v in avd_dict.items():
    dataset_dict[k + start_label_idx] = v

image = cv2.imread(f'{data_folder}/{scene}/jpg_rgb/{img_name}.jpg')
sseg_vote = cv2.imread(f'{label_folder}/{scene}/{img_name}_maskFormer_labels.png', cv2.IMREAD_UNCHANGED)
sseg_vote += 1

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
ax.imshow(image)
unique_labels = np.unique(sseg_vote)
unique_labels = np.delete(unique_labels, np.where(unique_labels == 0))
for label in unique_labels:
    if label == 1500:
        continue
    binary_mask = (sseg_vote == label).astype(np.uint8)
    mask_color = COLOR[label % len(COLOR), 0:3]/255
    text = dataset_dict[label].split(',')[0]

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
# plt.show()
fig.savefig(f'{folder}/{img_name}_maskformer_labels.jpg')
plt.close()
