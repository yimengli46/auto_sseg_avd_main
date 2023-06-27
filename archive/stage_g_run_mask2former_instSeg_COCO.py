
from detectron2.projects.deeplab import add_deeplab_config
import sys
import os
import glob
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

sys.path.insert(1, os.path.join(sys.path[0], '..'))  # NOQA
from Mask2Former.mask2former import add_maskformer2_config  # NOQA

coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(
    "../Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = '../Mask2Former/model_weights/model_final_e5f453.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
predictor = DefaultPredictor(cfg)

# ================================= initialize predictions ===============================
data_folder = '/projects/kosecka/Datasets/AVD_annotation-main'
saved_folder = 'output/stage_g_mask2former_instSeg_coco_results'

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

        outputs = predictor(image)

        # Show panoptic/instance/semantic predictions:
        v = Visualizer(image[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(30, 10))
        ax[0].imshow(image[:, :, ::-1])
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].imshow(instance_result)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.title(f'avd')

        fig.savefig(f'{saved_folder}/{img_name}.jpg',
                    bbox_inches='tight', dpi=200)
        plt.close()
