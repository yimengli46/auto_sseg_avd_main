"""
This script reads pre-generated image features for cabinet segments from an entire scene, applies 
K-Means clustering to group them, assigns each image segment to a specific cluster, and then 
visualizes the clustering results.
"""

import _pickle as cPickle
import bz2
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from constants import colormap
from utils import show_mask


data_folder = 'data/AVD_annotation-main'
saved_folder = 'output/exp_2_annotate_cabinet_handle'
stage_a_result_folder = 'output/stage_a_Detic_results'
stage_e_result_folder = 'output/stage_e_sam_dense_grid_prompts_results'

COLOR = colormap(rgb=True)
num_classes = 3


scene_list = ['Home_004_1', 'Home_005_1', 'Home_008_1']
# scene_list = [scene_list[0]]

# run it on existing images
for scene in scene_list:
    print(f'scene = {scene}')
    # load saved features
    pbz_list = [os.path.splitext(os.path.basename(x))[0]
                for x in sorted(glob.glob(f'{saved_folder}/selected_images/{scene}/*.pbz2'))]
    all_features = np.zeros((0, 2048)).astype(np.float32)
    for pbz_file in pbz_list:
        with bz2.BZ2File(f'{saved_folder}/selected_images/{scene}/{pbz_file}.pbz2', 'rb') as fp:
            segment_dict = cPickle.load(fp)
            batch_feature = segment_dict['segment_feature']
            all_features = np.vstack((all_features, batch_feature))

    # kmeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(all_features)

    img_name_list = [x[:15] for x in pbz_list]

    for img_name in img_name_list:
        print(f'img_name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/selected_images/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # =============== load SAM dense segments =======================
        sseg_sam = cv2.imread(
            f'{stage_e_result_folder}/selected_images/{img_name}_sam_segments.png', cv2.IMREAD_UNCHANGED)

        # go through each segment in sseg_sam
        segment_ids = np.unique(sseg_sam)

        with bz2.BZ2File(f'{saved_folder}/selected_images/{scene}/{img_name}_resnet_feature.pbz2', 'rb') as fp:
            segment_dict = cPickle.load(fp)
            wanted_segment_ids = segment_dict['segment_id']
            all_boxes = segment_dict['segment_bbox']
            batch_feature = segment_dict['segment_feature']

        # run the trained KMeans cluster
        segment_center_ids = kmeans.predict(batch_feature)

        # visualization bbox and mask
        H, W = sseg_sam.shape
        vis_mask = np.zeros((H, W, 3))

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 30))
        ax[0].imshow(image)
        for idx, segment_idx in enumerate(wanted_segment_ids):
            class_id = segment_center_ids[idx]
            class_color = COLOR[class_id % len(COLOR), 0:3]/255

            # print(f'mask.shape = {mask.shape}')
            mask = (sseg_sam == segment_idx)

            show_mask(mask, ax[0], color=class_color)
            vis_mask[sseg_sam == segment_idx] = class_color

        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].imshow(vis_mask)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{saved_folder}/selected_images/{scene}/{img_name}_clustering.jpg')
        plt.close()
