import _pickle as cPickle
import bz2
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
# from constants import colormap
from sklearn.manifold import TSNE


def show_mask(mask, ax, color):
    color = np.concatenate([color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


data_folder = 'data/AVD_annotation-main'
saved_folder = '/home/yimeng/ARGO_scratch/auto_sseg_avd/sseg_sam/output/exp_2_annotate_cabinet_handle'
stage_a_result_folder = 'output/stage_a_Detic_results'
stage_e_result_folder = 'output/stage_e_sam_dense_grid_prompts_results'

# COLOR = colormap(rgb=True)
# num_classes = 3


scene_list = ['Home_005_1']
scene_list = [scene_list[0]]

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

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(all_features)

    cluster_labels = kmeans.labels_

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Features')
    plt.show()

    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 30))
    # ax[0].imshow(image)

    # ax[0].get_xaxis().set_visible(False)
    # ax[0].get_yaxis().set_visible(False)
    # ax[1].imshow(vis_mask)
    # ax[1].get_xaxis().set_visible(False)
    # ax[1].get_yaxis().set_visible(False)
    # fig.tight_layout()
    # fig.savefig(f'{saved_folder}/selected_images/{scene}/{img_name}_clustering.jpg')
    # plt.close()
