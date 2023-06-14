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


# constants
WINDOW_NAME = "MaskFormer demo"


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

    # assert 1 == 2
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # run on ADE20K
    '''
    data_folder = '/projects/kosecka/Datasets/ADE20K/Semantic_Segmentation'
    saved_folder = 'output/ade20k_maskformer_results'

    # ============================== deal with OOD datasets ==================================
    img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

    for idx in range(img_list.shape[0]):
        img_dir = img_list[idx]['img']
        name = img_dir[18:-4]
        print(f'name = {name}')
        path = f'{data_folder}/{img_dir}'

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

        out_filename = f'{saved_folder}/{name}_mask.jpg'
        visualized_output.save(out_filename)

        # save predictions
        preds = predictions['sem_seg'].cpu().numpy()  # 150 x h x w
        sseg_img = np.argmax(preds, axis=0)  # h x w
        np.save(f'{saved_folder}/{name}.npy', sseg_img)
    '''

    # run on AVD
    data_folder = '/projects/kosecka/Datasets/AVD_annotation-main'
    saved_folder = 'output/AVD_maskFormer_results'

    scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
                  'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
                  'Home_015_1', 'Home_016_1',]

    for scene in scene_list:
        img_name_list = [os.path.splitext(os.path.basename(x))[0]
                         for x in sorted(glob.glob(f'{data_folder}/{scene}/selected_images/*.jpg'))]

        for img_name in img_name_list:
            print(f'name = {img_name}')
            path = f'{data_folder}/{scene}/selected_images/{img_name}.jpg'

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

            out_filename = f'{saved_folder}/{img_name}_mask.jpg'
            visualized_output.save(out_filename)

            # save predictions
            preds = predictions['sem_seg'].cpu().numpy()  # 150 x h x w
            sseg_img = np.argmax(preds, axis=0)  # h x w
            np.save(f'{saved_folder}/{img_name}.npy', sseg_img)
