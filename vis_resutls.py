'''
visualize segmentation results of ade20k and AVD dataset
a. original image
b. MaskFormer Semantic Segmentation results
c. SAM Segmentation results
d. SAM Segmentation with voting using MaskFormer labels results
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

'''
saved_folder = 'output/comparison_results_ADE20K'
data_folder = '/home/yimeng/ARGO_datasets/Datasets/ADE20K/Semantic_Segmentation'
sam_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/ade20k_sam_results'
sam_vote_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/vote_sam_with_maskFormer_results'
maskFormer_results_folder = '/home/yimeng/ARGO_scratch/sseg/MaskFormer/output/ade20k_maskformer_results'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

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
    ax[0][1].set_title(f"maskFormer")
    ax[1][0].imshow(vis_sam)
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title(f"SAM")
    ax[1][1].imshow(vis_sseg_vote)
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_title(f"SAM vote")
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'{saved_folder}/{name}.jpg')
    plt.close()

'''

saved_folder = 'output/comparison_results_AVD'
data_folder = '/home/yimeng/ARGO_datasets/Datasets/AVD_annotation-main'
sam_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/AVD_sam_results'
sam_vote_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/AVD_vote_sam_with_maskFormer_results'
maskFormer_results_folder = '/home/yimeng/ARGO_scratch/sseg/sseg_sam/output/AVD_maskFormer_results'

scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
              'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
              'Home_015_1', 'Home_016_1',]


for scene in scene_list:
    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/selected_images/*.jpg'))]

    for name in img_name_list:
        print(f'name = {name}')
        # load maskFormer results
        sseg_maskFormer = np.load(f'{maskFormer_results_folder}/{name}.npy', allow_pickle=True)
        sseg_maskFormer += 1
        vis_maskFormer = cv2.imread(f'{maskFormer_results_folder}/{name}_mask.jpg')
        vis_maskFormer = cv2.cvtColor(vis_maskFormer, cv2.COLOR_BGR2RGB)

        # load vote results
        sseg_vote = np.load(f'{sam_vote_results_folder}/{name}.npy', allow_pickle=True)
        sseg_vote += 1

        # load the image
        image = cv2.imread(f'{data_folder}/{scene}/selected_images/{name}.jpg')
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

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 15))
        ax[0][0].imshow(image)
        ax[0][0].get_xaxis().set_visible(False)
        ax[0][0].get_yaxis().set_visible(False)
        ax[0][0].set_title("rgb")
        ax[0][1].imshow(vis_maskFormer)
        ax[0][1].get_xaxis().set_visible(False)
        ax[0][1].get_yaxis().set_visible(False)
        ax[0][1].set_title(f"maskFormer")
        ax[1][0].imshow(vis_sam)
        ax[1][0].get_xaxis().set_visible(False)
        ax[1][0].get_yaxis().set_visible(False)
        ax[1][0].set_title(f"SAM")
        ax[1][1].imshow(vis_sseg_vote)
        ax[1][1].get_xaxis().set_visible(False)
        ax[1][1].get_yaxis().set_visible(False)
        ax[1][1].set_title(f"SAM vote")
        fig.tight_layout()
        # plt.show()
        fig.savefig(f'{saved_folder}/{name}.jpg')
        plt.close()
