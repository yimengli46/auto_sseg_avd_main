'''
stage f:
merge the outputs from Detic, MaskFormer and AVD instances.
'''
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from constants import ade20k_dict, lvis_dict, avd_dict

_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_LARGE_MASK_AREA_THRESH = 120000


def draw_text(
    ax,
    text,
    position,
    *,
    font_size=None,
    color="g",
    horizontal_alignment="center",
    rotation=0
):
    """
    Args:
        text (str): class label
        position (tuple): a tuple of the x and y coordinates to place text on image.
        font_size (int, optional): font of the text. If not provided, a font size
            proportional to the image width is calculated and used.
        color: color of the text. Refer to `matplotlib.colors` for full list
            of formats that are accepted.
        horizontal_alignment (str): see `matplotlib.text.Text`
        rotation: rotation angle in degrees CCW

    Returns:
        output (VisImage): image object with text drawn.
    """
    if not font_size:
        font_size = 10

    # since the text background is dark, we don't want the text to be dark
    # color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    # color[np.argmax(color)] = max(0.8, np.max(color))

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size,
        family="sans-serif",
        bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )
    # return self.output


def draw_binary_mask(ax, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5):
    """
    Args:
        binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
            W is the image width. Each value in the array is either a 0 or 1 value of uint8
            type.
        color: color of the mask. Refer to `matplotlib.colors` for a full list of
            formats that are accepted. If None, will pick a random color.
        edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
            full list of formats that are accepted.
        text (str): if None, will be drawn in the object's center of mass.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        area_threshold (float): a connected component small than this will not be shown.

    Returns:
        output (VisImage): image object with mask drawn.
    """

    # if color is None:
    #     color = random_color(rgb=True, maximum=1)
    # color = mplc.to_rgb(color)

    binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
    shape2d = (binary_mask.shape[0], binary_mask.shape[1])

    rgba = np.zeros(shape2d + (4,), dtype="float32")
    rgba[:, :, :3] = color
    rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
    ax.imshow(rgba)  # , extent=(0, , self.output.height, 0))

    if text is not None:
        # TODO sometimes drawn on wrong objects. the heuristics here can improve.
        # lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
        lighter_color = (1, 1, 1)
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
        largest_component_id = np.argmax(stats[1:, -1]) + 1

        # draw text on the largest component, as well as other very large components.
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                # median is more stable than centroid
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                draw_text(ax, text, center, color=lighter_color)


# merge dataset dicts
dataset_dict = {}
# merge with ade20k
start_label_idx = 0
for k, v in ade20k_dict.items():
    dataset_dict[k + start_label_idx] = v
# merge with lvis
start_label_idx = 150
for k, v in lvis_dict.items():
    dataset_dict[k + start_label_idx] = v
# assert 1 == 2
# merge with avd instance
start_label_idx = 1500
for k, v in avd_dict.items():
    dataset_dict[k + start_label_idx] = v


saved_folder = 'output/stage_f_merge_results'
stage_b_results_folder = 'output/stage_b_sam_results'
stage_c_results_folder = 'output/stage_c_sam_results_with_avd_instances'
stage_d_results_folder = 'output/stage_d_maskFormer_results'
stage_e_results_folder = 'output/stage_e_sam_dense_grid_prompts_results'

data_folder = '/projects/kosecka/Datasets/AVD_annotation-main'

scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
              'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
              'Home_015_1', 'Home_016_1']
# scene_list = [scene_list[0]]

for scene in scene_list:
    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/selected_images/*.jpg'))]
    # img_name_list = img_name_list[:3]

    for img_name in img_name_list:

        print(f'name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/selected_images/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load sam dense grid prompt results
        sseg_sam = cv2.imread(f'{stage_e_results_folder}/{img_name}_sam_segments.png', cv2.IMREAD_UNCHANGED)

        H, W = sseg_sam.shape
        sseg_vote = np.zeros((H, W), dtype=np.uint16)

        # ==================================== merge with maskFormer results
        # label index 0 is the background
        start_label_idx = 0 + 1
        # load maskFormer results
        sseg_maskFormer = cv2.imread(f'{stage_d_results_folder}/{img_name}_maskFormer_labels.png', cv2.IMREAD_UNCHANGED)

        # go through each segment in sseg_sam
        segment_ids = np.unique(sseg_sam)
        for segment_id in segment_ids:
            mask = (sseg_sam == segment_id)
            # get the segment from maskFormer result
            segment = sseg_maskFormer[mask]
            counts = np.bincount(segment)
            most_common_idx = np.argmax(counts)
            sseg_vote[mask] = most_common_idx

        sseg_vote += start_label_idx

        # ==================================== merge with Detic result
        # label index 0 is the background
        start_label_idx = 150 + 1

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
        fig.savefig(f'{saved_folder}/{img_name}_sseg.jpg')
        plt.close()

        cv2.imwrite(f'{saved_folder}/{img_name}_labels.png', sseg_vote)
