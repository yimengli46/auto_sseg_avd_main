"""
This script is for evaluation on ADE20K.
It runs Detic on ADE20K indoor images.
"""


from detectron2.config import get_cfg
import sys
import cv2
import os
import numpy as np
import bz2
import _pickle as cPickle
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '../auto_sseg_avd_Detic/third_party/CenterNet2/'))
sys.path.insert(2, os.path.join(sys.path[0], '../auto_sseg_avd_Detic'))  # NOQA
from centernet.config import add_centernet_config  # NOQA
from detic.predictor import VisualizationDemo  # NOQA
from detic.config import add_detic_config  # NOQA


class Args:
    def __init__(self, vocabulary, custom_vocabulary=''):
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

# args = Args('lvis')
custom_vocabulary = 'person,plant,painting,lamp,bathtub,cushion,sink,pillow,toilet,chandelier,tv,bottle,bag,oven,ball,microwave,flowerpot,bicycle,sculpture,vase,trash can,plate,shower,drinking glass,clock,flag'
args = Args('custom', custom_vocabulary)
demo = VisualizationDemo(cfg, args)  # , category_str)

# ================================= initialize predictions ===============================
data_folder = '/projects/kosecka/Datasets/ADE20K/Semantic_Segmentation'
saved_folder = 'output/ade20k_Detic_results'

img_list = np.load(f'{data_folder}/val_img_list.npy', allow_pickle=True)

for idx in range(img_list.shape[0]):
    img_dir = img_list[idx]['img']
    img_name = img_dir[18:-4]
    print(f'name = {img_name}')

    image = cv2.imread(f'{data_folder}/{img_dir}')
    # assert 1 == 2

    predictions, visualized_output = demo.run_on_image(image)
    panor_panopSeg = visualized_output.get_image()

    # ========================== save the detection results ============
    predictions_dict = {}
    num_predictions = len(predictions['instances'])
    print(f'num_instances = {num_predictions}')
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

    fig.savefig(f'{saved_folder}/{img_name}_vis_Detic.jpg',
                bbox_inches='tight', dpi=200)
    plt.close()
