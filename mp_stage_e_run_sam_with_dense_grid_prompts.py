'''
stage e of the auto-labeling process:
run SAM to get the raw segments without semantic predictions.
'''
import multiprocessing
import skimage.measure
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
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


def run_sam__with_dense_grid_prompts(scene):
    print(f'scene = {scene}')

    sam_checkpoint = "../segment-anything/model_weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device_id = gpu_Q.get()
    device = f"cuda:{device_id}"
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

    # run on AVD
    data_folder = '/projects/kosecka/Datasets/ActiveVisionDataset'
    saved_folder = 'output/stage_e_sam_dense_grid_prompts_results'
    scene_folder = f'{saved_folder}/{scene}'
    if not os.path.exists(scene_folder):
        os.mkdir(scene_folder)

    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/jpg_rgb/*.jpg'))]
    print(f'==> {len(img_name_list)} images to process ...')

    for img_name in img_name_list:
        print(f'img_name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/jpg_rgb/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        masks = mask_generator.generate(image)

        img_mask = np.zeros((H, W), dtype=np.uint16)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        count_mask = 1
        for ann in sorted_masks:
            m = ann['segmentation']
            # color_mask = np.random.random(3)
            img_mask[m] = count_mask
            count_mask += 1

        assert count_mask - 1 == len(sorted_masks)

        instance_label, num_ins = skimage.measure.label(
            img_mask == 0, background=0, connectivity=1, return_num=True)

        for idx_ins in range(1, num_ins + 1):
            img_mask[instance_label == idx_ins] = count_mask
            count_mask += 1

        # reformat the labels
        unique_labels = np.unique(img_mask)

        vis_mask = np.ones((H, W, 3))
        for idx in unique_labels:
            vis_mask[img_mask == idx] = np.random.random(3)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
        ax.imshow(vis_mask)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{scene_folder}/{img_name}_mask.jpg')
        plt.close()

        # print(f'min_mask = {img_mask.min()}')

        assert img_mask.min() > 0

        img_mask = img_mask.astype(np.uint16)
        cv2.imwrite(f'{scene_folder}/{img_name}_sam_segments.png', img_mask)

    gpu_Q.put(device_id)
    return


def mp_run_wrapper(args):
    run_sam__with_dense_grid_prompts(args[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--j', type=int, required=False, default=1)
    args = parser.parse_args()

    scene_list = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2', 'Home_004_1',
                  'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
                  'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_015_1',
                  'Home_016_1', 'Office_001_1']

    # ====================== get the available GPU devices ============================
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    devices = [int(dev) for dev in visible_devices]

    for device_id in devices:
        for _ in range(args.j):
            gpu_Q.put(device_id)

    with multiprocessing.Pool(processes=8) as pool:
        args0 = scene_list
        pool.map(mp_run_wrapper, list(zip(args0)))
        pool.close()


if __name__ == "__main__":
    gpu_Q = multiprocessing.Queue()
    main()
