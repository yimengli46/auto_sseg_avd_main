from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
import mss
import sys
import tqdm
import cv2
import warnings
import time
import tempfile
import os
import numpy as np
import multiprocessing as mp
import glob
import argparse

import json
import bz2
import _pickle as cPickle
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '../auto_sseg_avd_Detic/third_party/CenterNet2/'))
sys.path.insert(2, os.path.join(sys.path[0], '../auto_sseg_avd_Detic'))  # NOQA
from centernet.config import add_centernet_config  # NOQA
from detic.predictor import VisualizationDemo  # NOQA
from detic.config import add_detic_config  # NOQA


class Args:
    def __init__(self, vocabulary, custom_vocabulary=[]):
        self.vocabulary = vocabulary
        self.custom_vocabulary = custom_vocabulary


config_file = '../auto_sseg_avd_Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
config_file = 'configs/my_Detic_config.yaml'
confidence_threshold = 0.5
opts = ['MODEL.WEIGHTS',
        '../auto_sseg_avd_Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth',
        'MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH',
        '../auto_sseg_avd_Detic/datasets/metadata/lvis_v1_train_cat_info.json',
        'MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH',
        '../auto_sseg_avd_Detic/datasets/metadata/lvis_v1_clip_a+cname.npy']

# ========================================= setup cfg ====================================
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(config_file)
cfg.merge_from_list(opts)
# Set score_threshold for builtin models
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
if True:
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
cfg.freeze()

args = Args('coco')
demo = VisualizationDemo(cfg, args)  # , category_str)

# ================================= initialize predictions ===============================
data_folder = '/projects/kosecka/Datasets/AVD_annotation-main'
saved_folder = 'output/stage_g_Detic_coco_results'

scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
              'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
              'Home_015_1', 'Home_016_1']
# scene_list = [scene_list[0]]

for scene in scene_list:
    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/selected_images/*.jpg'))]
    # img_name_list = img_name_list[:3]

    for img_name in img_name_list:
        print(f'img_name = {img_name}')
        image = cv2.imread(f'{data_folder}/{scene}/selected_images/{img_name}.jpg')

        predictions, visualized_output = demo.run_on_image(image)
        panor_panopSeg = visualized_output.get_image()

        # ========================== save the detection results ============
        predictions_dict = {}
        predictions_dict['num_instances'] = len(predictions['instances'])
        predictions_dict['pred_boxes'] = predictions['instances'].pred_boxes.tensor.cpu(
        ).numpy()
        predictions_dict['scores'] = predictions['instances'].scores.cpu(
        ).numpy()
        predictions_dict['pred_classes'] = predictions['instances'].pred_classes.cpu(
        ).numpy()
        predictions_dict['pred_masks'] = predictions['instances'].pred_masks.cpu(
        ).numpy()
        with bz2.BZ2File(f'{saved_folder}/{img_name}.pbz2', 'w') as fp:
            cPickle.dump(
                predictions_dict,
                fp
            )

        # ======================== visualization =========================
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(30, 10))
        ax[0].imshow(image[:, :, ::-1])
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].imshow(panor_panopSeg)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.title(f'avd')

        fig.savefig(f'{saved_folder}/{img_name}.jpg',
                    bbox_inches='tight', dpi=200)
        plt.close()
