'''
stage f generate labels for instance segmentation:
merge the outputs from Detic and AVD instances.
'''
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from constants import ade20k_dict, lvis_dict, avd_dict
from utils import _OFF_WHITE, draw_text, draw_binary_mask


# merge dataset dicts
dataset_dict = {}
# merge with lvis
start_label_idx = 0
for k, v in lvis_dict.items():
    dataset_dict[k + start_label_idx] = v
# merge with avd instance
start_label_idx = 1500
for k, v in avd_dict.items():
    dataset_dict[k + start_label_idx] = v


saved_folder = 'output/stage_f_inst_seg'
stage_a_results_folder = 'output/stage_a_Detic_results/selected_images'
stage_b_results_folder = 'output/stage_b_sam_results/selected_images'
stage_c_results_folder = 'output/stage_c_sam_results_with_avd_instances/selected_images'

data_folder = 'data/AVD_annotation-main'

scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
              'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
              'Home_015_1', 'Home_016_1']
scene_list = [scene_list[0]]

for scene in scene_list:
    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/selected_images/*.jpg'))]
    img_name_list = img_name_list[:3]

    for img_name in img_name_list:

        print(f'name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/selected_images/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]
        sseg_vote = np.zeros((H, W), dtype=np.uint16)

        # ==================================== merge with Detic result
        # label index 0 is the background
        start_label_idx = 1

        # load SAM_Detic result
        sseg_Detic = cv2.imread(f'{stage_b_results_folder}/{img_name}_mask_labels.png', cv2.IMREAD_UNCHANGED)
        mask = (sseg_Detic > 0)
        sseg_Detic[mask] += start_label_idx

        sseg_vote = np.where(mask, sseg_Detic, sseg_vote)

        # ================================ merge with AVD instance result
        start_label_idx = 1500

        # load SAM avd instance result (some images do not have AVD instances)
        try:
            sseg_avd = cv2.imread(f'{stage_c_results_folder}/{img_name}_avd_instances_labels.png', cv2.IMREAD_UNCHANGED)
            mask = (sseg_avd > 0)
            sseg_avd[mask] += start_label_idx

            sseg_vote = np.where(mask, sseg_avd, sseg_vote)
        except:
            print('no avd instance in this image.')

        # ================= visualization
        # vis_sseg_vote = np.ones((H, W, 3))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
        ax.imshow(image)
        unique_labels = np.unique(sseg_vote)
        unique_labels = np.delete(unique_labels, np.where(unique_labels == 0))
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
        plt.show()
        # fig.savefig(f'{saved_folder}/{img_name}_sseg.jpg')
        # plt.close()

        # cv2.imwrite(f'{saved_folder}/{img_name}_labels.png', sseg_vote)
