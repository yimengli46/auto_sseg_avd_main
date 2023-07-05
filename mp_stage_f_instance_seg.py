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
from constants import ade20k_dict, lvis_dict, avd_dict, UNWANTED_CLASSES, ALLOWED_OBJECT_OVERLAY_PAIRS
from utils import _OFF_WHITE, draw_binary_mask, comp_bbox_iou, comp_mask_iou
import _pickle as cPickle
import bz2
import multiprocessing

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
# merge with avd instance
start_label_idx = 1500
for k, v in avd_dict.items():
    dataset_dict[k + start_label_idx] = v


def run_instance_segmentation(scene):
    print(f'scene = {scene}')

    saved_folder = 'output/stage_f_inst_seg'
    scene_folder = f'{saved_folder}/{scene}'
    if not os.path.exists(scene_folder):
        os.mkdir(scene_folder)
    stage_a_result_folder = f'output/stage_a_Detic_results/{scene}'
    stage_b_result_folder = f'output/stage_b_sam_results/{scene}'
    stage_c_result_folder = f'output/stage_c_sam_results_with_avd_instances/{scene}'

    data_folder = 'data/ActiveVisionDataset'

    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/jpg_rgb/*.jpg'))]
    # img_name_list = img_name_list[:4]

    for img_name in img_name_list:

        print(f'name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/jpg_rgb/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]
        sseg_vote = np.zeros((H, W), dtype=np.uint16)

        # ==================================== merge with Detic result
        try:
            # label index 0 is the background
            start_label_idx = 150 + 1

            # load Detic boxes
            with bz2.BZ2File(f'{stage_a_result_folder}/{img_name}.pbz2', 'rb') as fp:
                pred_dict = cPickle.load(fp)
                num_instances = pred_dict['num_instances']
                detic_pred_boxes = pred_dict['pred_boxes'].astype(np.int32)
                scores = pred_dict['scores']
                detic_pred_classes = pred_dict['pred_classes']

            # load Detic-SAM masks
            with bz2.BZ2File(f'{stage_b_result_folder}/{img_name}_masks.pbz2', 'rb') as fp:
                detic_masks = cPickle.load(fp)

            # step : remove unwanted classes
            idx_mask = np.isin(detic_pred_classes + start_label_idx, UNWANTED_CLASSES, invert=True)
            detic_pred_boxes = detic_pred_boxes[idx_mask]
            detic_pred_classes = detic_pred_classes[idx_mask]
            detic_masks = detic_masks[idx_mask]
            scores = scores[idx_mask]
            num_instances = idx_mask.sum()

            # '''
            # step : remove Detic masks overlaps with AVD instances
            flag_avd_exists = False
            try:
                with bz2.BZ2File(f'{stage_c_result_folder}/{img_name}_avd_instances_masks.pbz2', 'rb') as fp:
                    flag_avd_exists = True
                    avd_dict = cPickle.load(fp)
                    avd_pred_boxes = avd_dict['pred_boxes'].astype(np.int32)
                    avd_pred_classes = pred_dict['pred_classes']

                detic_bbox_idx_arr = np.array(range(detic_pred_boxes.shape[0]), dtype=int)
                unwanted_bbox_idx_list = []
                for (i_avd, avd_box) in enumerate(avd_pred_boxes):
                    for (i_detic, detic_box) in enumerate(detic_pred_boxes):
                        # if IoU > thresh, put in unwanted list
                        iou = comp_bbox_iou(avd_box, detic_box)
                        if iou > 0.5:
                            # print(f'iou = {iou}')
                            unwanted_bbox_idx_list.append(i_detic)

                # remove bbox from detic box and detic mask and detic classes.
                for idx in unwanted_bbox_idx_list:
                    mask_class = detic_pred_classes[idx] + start_label_idx
                    # print(f'unwanted id = {mask_class}, class = {dataset_dict[mask_class]}')

                idx_mask = np.isin(detic_bbox_idx_arr, unwanted_bbox_idx_list, invert=True)
                detic_pred_boxes = detic_pred_boxes[idx_mask]
                detic_pred_classes = detic_pred_classes[idx_mask]
                detic_masks = detic_masks[idx_mask]
                scores = scores[idx_mask]
                num_instances = idx_mask.sum()
            except:
                print('no avd instance in this image.')
            # '''

            # sort the masks ascendingly. So the small masks are in the front
            # '''
            sorted_masks = sorted(enumerate(detic_masks), key=(lambda x: x[1].sum()), reverse=False)
            sorted_masks_idx, _ = zip(*sorted_masks)
            sorted_masks_idx = np.array(sorted_masks_idx)
            detic_pred_boxes = detic_pred_boxes[sorted_masks_idx]
            detic_pred_classes = detic_pred_classes[sorted_masks_idx]
            detic_masks = detic_masks[sorted_masks_idx]
            scores = scores[sorted_masks_idx]
            # '''

            # step : remove the smaller mask with large mask iou with a larger mask.
            # '''
            unwanted_mask_idx_list = []
            detic_mask_idx_arr = list(range(detic_masks.shape[0]))
            for idx_mask_a in detic_mask_idx_arr[:-1]:
                for idx_mask_b in detic_mask_idx_arr[idx_mask_a + 1:]:

                    mask_a = detic_masks[idx_mask_a]
                    mask_b = detic_masks[idx_mask_b]
                    # if IoU > thresh, put in unwanted list
                    iou = comp_mask_iou(mask_a, mask_b)
                    if iou > 0.5:
                        # # step : check if the overlay between obj_a and obj_b is allowed
                        # class_a = dataset_dict[detic_pred_classes[idx_mask_a] + start_label_idx]
                        # class_b = dataset_dict[detic_pred_classes[idx_mask_b] + start_label_idx]

                        # if class_a in ALLOWED_OBJECT_OVERLAY_PAIRS and class_b in ALLOWED_OBJECT_OVERLAY_PAIRS[class_a]:
                        #     print(f'class_a = {class_a}, class_b = {class_b}')
                        #     pass
                        # else:
                        #     unwanted_mask_idx_list.append(idx_mask_a)
                        unwanted_mask_idx_list.append(idx_mask_a)

            idx_mask = np.isin(np.array(range(num_instances), dtype=int), unwanted_mask_idx_list, invert=True)
            detic_pred_boxes = detic_pred_boxes[idx_mask]
            detic_pred_classes = detic_pred_classes[idx_mask]
            detic_masks = detic_masks[idx_mask]
            scores = scores[idx_mask]
            num_instances = idx_mask.sum()
            # '''

            # convert all the masks into one simgle mask
            Detic_mask = np.zeros((H, W), dtype=np.uint16)
            for idx_mask in range(detic_masks.shape[0]):
                mask = detic_masks[idx_mask]
                mask_class = detic_pred_classes[idx_mask] + start_label_idx
                # print(f'id = {mask_class}, class = {dataset_dict[mask_class]}')
                Detic_mask[mask] = mask_class

            sseg_vote = np.where(Detic_mask > 0, Detic_mask, sseg_vote)
        except:
            print('no Detic objects in this image.')

        # ================================ merge with AVD instance result
        # '''
        start_label_idx = 1500

        # load SAM avd instance result (some images do not have AVD instances)
        try:
            sseg_avd = cv2.imread(
                f'{stage_c_result_folder}/{img_name}_avd_instances_labels.png', cv2.IMREAD_UNCHANGED)
            mask = (sseg_avd > 0)
            sseg_avd[mask] += start_label_idx

            sseg_vote = np.where(mask, sseg_avd, sseg_vote)
        except:
            print('no avd instance in this image.')
        # '''

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
        # plt.show()
        fig.savefig(f'{scene_folder}/{img_name}_sseg.jpg')
        plt.close()

        cv2.imwrite(f'{scene_folder}/{img_name}_labels.png', sseg_vote)


def mp_run_wrapper(args):
    run_instance_segmentation(args[0])


def main():
    scene_list = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2', 'Home_004_1',
                  'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
                  'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_015_1',
                  'Home_016_1', 'Office_001_1']

    with multiprocessing.Pool(processes=20) as pool:
        args0 = scene_list
        pool.map(mp_run_wrapper, list(zip(args0)))
        pool.close()


if __name__ == "__main__":
    main()
