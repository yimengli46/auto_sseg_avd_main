'''
This script reads the SAM segmentation results and MaskFormer semantic segmentation predictions.
It determines the category of SAM segments by counting the number of pixels for each category
in the MaskFormer predictions and selecting the class with the highest vote. The experiment is
conducted on the AVD and ADE20K datasets.
'''
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
saved_folder = 'output/vote_sam_with_maskFormer_results'
sam_results_folder = 'output/ade20k_sam_results'
maskFormer_results_folder = '../MaskFormer/output/ade20k_maskformer_results'
data_folder = '/projects/kosecka/Datasets/ADE20K/Semantic_Segmentation'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['img']
    name = img_dir[18:-4]
    print(f'name = {name}')

    # load sam results
    sseg_sam = np.load(f'{sam_results_folder}/{name}.npy', allow_pickle=True)

    # load maskFormer results
    sseg_maskFormer = np.load(f'{maskFormer_results_folder}/{name}.npy', allow_pickle=True)

    H, W = sseg_sam.shape
    sseg_vote = np.zeros((H, W), dtype=np.int32)
    # go through each segment in sseg_sam
    segment_ids = np.unique(sseg_sam)
    for segment_id in segment_ids:
        mask = (sseg_sam == segment_id)
        # get the segment from maskFormer result
        segment = sseg_maskFormer[mask]
        counts = np.bincount(segment)
        most_common_idx = np.argmax(counts)
        sseg_vote[mask] = most_common_idx

    vis_sseg_vote = np.ones((H, W, 3))
    unique_labels = np.unique(sseg_vote)
    for idx in unique_labels:
        vis_sseg_vote[sseg_vote == idx] = np.random.random(3)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    ax.imshow(vis_sseg_vote)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{saved_folder}/{name}_vote_sseg.jpg')
    plt.close()

    np.save(f'{saved_folder}/{name}.npy', sseg_vote)
'''

saved_folder = 'output/AVD_vote_sam_with_maskFormer_results'
sam_results_folder = 'output/AVD_sam_results'
maskFormer_results_folder = 'output/AVD_maskFormer_results'
data_folder = '/projects/kosecka/Datasets/AVD_annotation-main'

scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
              'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
              'Home_015_1', 'Home_016_1',]


for scene in scene_list:
    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/selected_images/*.jpg'))]

    for img_name in img_name_list:

        print(f'name = {img_name}')

        # load sam results
        sseg_sam = np.load(f'{sam_results_folder}/{img_name}.npy', allow_pickle=True)

        # load maskFormer results
        sseg_maskFormer = np.load(f'{maskFormer_results_folder}/{img_name}.npy', allow_pickle=True)

        H, W = sseg_sam.shape
        sseg_vote = np.zeros((H, W), dtype=np.int32)
        # go through each segment in sseg_sam
        segment_ids = np.unique(sseg_sam)
        for segment_id in segment_ids:
            mask = (sseg_sam == segment_id)
            # get the segment from maskFormer result
            segment = sseg_maskFormer[mask]
            counts = np.bincount(segment)
            most_common_idx = np.argmax(counts)
            sseg_vote[mask] = most_common_idx

        vis_sseg_vote = np.ones((H, W, 3))
        unique_labels = np.unique(sseg_vote)
        for idx in unique_labels:
            vis_sseg_vote[sseg_vote == idx] = np.random.random(3)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
        ax.imshow(vis_sseg_vote)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{saved_folder}/{img_name}_vote_sseg.jpg')
        plt.close()

        np.save(f'{saved_folder}/{img_name}.npy', sseg_vote)
