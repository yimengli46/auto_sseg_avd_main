'''
run maskFormer on ade20k and AVD dataset
'''
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # NOQA
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from MaskFormer.mask_former import add_mask_former_config
from MaskFormer.demo.predictor import VisualizationDemo

import matplotlib.pyplot as plt
from constants import colormap
from utils import _OFF_WHITE, draw_text, draw_binary_mask
from constants import ade20k_dict
WINDOW_NAME = "MaskFormer demo"

COLOR = colormap(rgb=True)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    args.config_file = '../MaskFormer/configs/ade20k-150/swin/maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml'
    args.opts = ['MODEL.WEIGHTS', '../MaskFormer/model_weights/model_final_aefa3b.pkl']

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # run on AVD
    data_folder = '/projects/kosecka/Datasets/ActiveVisionDataset'
    saved_folder = 'output/stage_d_maskFormer_results'

    scene_list = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2', 'Home_004_1',
                  'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
                  'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_015_1',
                  'Home_016_1', 'Office_001_1']
    scene_list = [scene_list[7]]

    for scene in scene_list:
        print(f'scene = {scene}')
        scene_folder = f'{saved_folder}/{scene}'
        if not os.path.exists(scene_folder):
            os.mkdir(scene_folder)

        img_name_list = [os.path.splitext(os.path.basename(x))[0]
                         for x in sorted(glob.glob(f'{data_folder}/{scene}/jpg_rgb/*.jpg'))]

        for img_name in img_name_list[70:]:
            print(f'name = {img_name}')
            path = f'{data_folder}/{scene}/jpg_rgb/{img_name}.jpg'

            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            out_filename = f'{scene_folder}/{img_name}_mask.jpg'
            visualized_output.save(out_filename)

            # save predictions
            preds = predictions['sem_seg'].cpu().numpy()  # 150 x h x w
            sseg_img = np.argmax(preds, axis=0)  # h x w

            sseg_img = sseg_img.astype(np.uint16)
            cv2.imwrite(f'{scene_folder}/{img_name}_maskFormer_labels.png', sseg_img)

            sseg_img += 1
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
            ax.imshow(img[:, :, ::-1])
            unique_labels = list(np.unique(sseg_img))
            if 0 in unique_labels:
                unique_labels = unique_labels[1:]

            for label in unique_labels:
                binary_mask = (sseg_img == label).astype(np.uint8)
                mask_color = COLOR[label % len(COLOR), 0:3]/255
                text = ade20k_dict[label]
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
            fig.savefig(f'{scene_folder}/{img_name}_maskFormer_vis.jpg')
            plt.close()
