"""
This script does multiprocessing.
Each process reads pre-generated image features for cabinet segments from an entire scene, applies 
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


def show_mask(mask, ax, color):
    color = np.concatenate([color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


data_folder = 'data/ActiveVisionDataset'
saved_folder = 'output/exp_2_annotate_cabinet_handle'
stage_a_result_folder = 'output/stage_a_Detic_results'
stage_e_result_folder = 'output/stage_e_sam_dense_grid_prompts_results'

COLOR = colormap(rgb=True)
num_classes = 4

scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
              'Home_007_1', 'Home_008_1']

for scene in scene_list:
    print(f'scene = {scene}')
    scene_folder = f'{saved_folder}/{scene}'
    if not os.path.exists(scene_folder):
        os.mkdir(scene_folder)

    # load saved features
    pbz_list = [os.path.splitext(os.path.basename(x))[0]
                for x in sorted(glob.glob(f'{scene_folder}/*.pbz2'))]
    all_features = np.zeros((0, 2048)).astype(np.float32)
    for pbz_file in pbz_list:
        with bz2.BZ2File(f'{saved_folder}/{scene}/{pbz_file}.pbz2', 'rb') as fp:
            segment_dict = cPickle.load(fp)
            batch_feature = segment_dict['segment_feature']
            all_features = np.vstack((all_features, batch_feature))

    # kmeans clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init="auto").fit(all_features)
    print(f'finished kmeans clustering training with {all_features.shape[0]} datapoints ...')

    img_name_list = [x[:15] for x in pbz_list]

    for img_name in img_name_list:
        print(f'img_name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/jpg_rgb/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # =============== load SAM dense segments =======================
        sseg_sam = cv2.imread(f'{stage_e_result_folder}/{scene}/{img_name}_sam_segments.png', cv2.IMREAD_UNCHANGED)

        # go through each segment in sseg_sam
        segment_ids = np.unique(sseg_sam)

        with bz2.BZ2File(f'{saved_folder}/{scene}/{img_name}_resnet_feature.pbz2', 'rb') as fp:
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
        fig.savefig(f'{scene_folder}/{img_name}_clustering.jpg')
        plt.close()
