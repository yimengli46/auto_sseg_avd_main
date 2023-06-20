import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # NOQA
from GroundingDINO.groundingdino.util.inference import Model
import supervision as sv


def enhance_class_name(class_names):
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


GROUNDING_DINO_CONFIG_PATH = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "../GroundingDINO/model_weights/groundingdino_swint_ogc.pth"


grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SOURCE_IMAGE_PATH = f"/home/yimeng/ARGO_datasets/Datasets/AVD_annotation-main/Home_004_1/selected_images/000410000020101.jpg"
CLASSES = ['pot', 'glass', 'bottle', 'coffee machine']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


# load image
image = cv2.imread(SOURCE_IMAGE_PATH)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=enhance_class_name(class_names=CLASSES),
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

sv.plot_image(annotated_frame, (16, 16))
