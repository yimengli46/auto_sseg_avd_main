'''
run SAM on AVD with Detic detected bbox as prompt.
'''
from torchvision.ops import roi_align
from utils import compute_intersection_area_iou_to_box1
import torchvision.models as models
import torchvision.transforms as transforms
import _pickle as cPickle
import bz2
import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing
import argparse
import cv2
sys.path.append("..")


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def batch_segment_input_boxes(sam_predictor, image, boxes):
    H, W = image.shape[:2]
    sam_predictor.set_image(image)
    boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, (H, W))

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    masks = masks.cpu().numpy()

    return masks


def run_sam_with_boxes_prompts(scene):
    print(f'scene = {scene}')
    # =========================== initialize resnet model =================================
    device = "cuda"
    resnet50 = models.resnet50(pretrained=True).to(device)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])
    resnet50.eval()

    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ============================= run on AVD =======================================
    data_folder = 'data/ActiveVisionDataset'
    saved_folder = 'output/exp_2_annotate_cabinet_handle/'
    scene_folder = f'{saved_folder}/{scene}'
    if not os.path.exists(scene_folder):
        os.mkdir(scene_folder)
    stage_a_result_folder = 'output/stage_a_Detic_results'
    stage_e_result_folder = f'output/stage_e_sam_dense_grid_prompts_results/{scene}'

    img_name_list = [os.path.splitext(os.path.basename(x))[0]
                     for x in sorted(glob.glob(f'{data_folder}/{scene}/jpg_rgb/*.jpg'))]
    # img_name_list = img_name_list[:6]

    for img_name in img_name_list:
        print(f'img_name = {img_name}')

        image = cv2.imread(f'{data_folder}/{scene}/jpg_rgb/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        # get image coordinates
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)
        image_coords = np.stack((xv, yv), axis=2)

        # load Detic boxes
        with bz2.BZ2File(f'{stage_a_result_folder}/{scene}/{img_name}.pbz2', 'rb') as fp:
            pred_dict = cPickle.load(fp)
            num_instances = pred_dict['num_instances']
            pred_boxes = pred_dict['pred_boxes'].astype(np.int32)
            pred_classes = pred_dict['pred_classes']
            # pred_masks = pred_dict['pred_masks']

        # extract cabinet
        mask = (pred_classes == 180)
        pred_classes = pred_classes[mask]
        pred_boxes = pred_boxes[mask]

        if pred_boxes.shape[0] == 0:
            continue

        # ============= get resnet50 embedding ==============
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        input_batch = input_batch.to(device)

        with torch.no_grad():
            features = resnet50(input_batch)

        # =============== load SAM dense segments =======================
        sseg_sam = cv2.imread(f'{stage_e_result_folder}/{img_name}_sam_segments.png', cv2.IMREAD_UNCHANGED)

        H, W = sseg_sam.shape

        # go through each segment in sseg_sam
        wanted_segment_ids = []
        all_boxes = []
        segment_ids = np.unique(sseg_sam)
        for segment_id in segment_ids:
            mask = (sseg_sam == segment_id)

            # convert mask to a bounding box
            mask_coords = image_coords[mask]  # N x 2
            min_x = mask_coords[:, 0].min()
            max_x = mask_coords[:, 0].max()
            min_y = mask_coords[:, 1].min()
            max_y = mask_coords[:, 1].max()

            bbox_segment = [min_x, min_y, max_x, max_y]

            # check if mask points is inside a cabinet bbox
            flag = False
            for box_idx in range(pred_boxes.shape[0]):
                cabinet_bbox = pred_boxes[box_idx]
                iou = compute_intersection_area_iou_to_box1(bbox_segment, cabinet_bbox)
                if iou > 0.9:
                    # print(f'iou = {iou}')
                    flag = True
                    break

            if flag:
                wanted_segment_ids.append(segment_id)
                all_boxes.append(bbox_segment)

        # =============== extract segment features ==============
        all_boxes = np.array(all_boxes)  # N x 4
        N = all_boxes.shape[0]
        batch_boxes = torch.tensor(all_boxes).to(device).float()
        batch_feature = roi_align(features, [batch_boxes],
                                  output_size=(1, 1), spatial_scale=1/32.0, aligned=True)  # N x 2048 x 1 x 1
        batch_feature = batch_feature.view((N, 2048)).detach().cpu().numpy()

        # ============= save as pickle ===================
        segment_dict = {}
        segment_dict['segment_id'] = wanted_segment_ids
        segment_dict['segment_bbox'] = all_boxes
        segment_dict['segment_feature'] = batch_feature

        with bz2.BZ2File(f'{scene_folder}/{img_name}_resnet_feature.pbz2', 'w') as fp:
            cPickle.dump(
                segment_dict,
                fp
            )

        # visualization bbox and mask
        vis_mask = np.zeros((H, W, 3))
        for idx in wanted_segment_ids:
            vis_mask[sseg_sam == idx] = np.random.random(3)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 30))
        ax[0].imshow(image)
        for idx in wanted_segment_ids:
            # print(f'mask.shape = {mask.shape}')
            mask = (sseg_sam == idx)
            show_mask(mask, ax[0], random_color=True)
        # for box in pred_boxes:
        #    show_box(box, ax[0])
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].imshow(vis_mask)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{scene_folder}/{img_name}_vis.jpg')
        plt.close()

    return


def mp_run_wrapper(args):
    run_sam_with_boxes_prompts(args[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--j', type=int, required=False, default=1)
    args = parser.parse_args()

    scene_list = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2', 'Home_004_1',
                  'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
                  'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_015_1',
                  'Home_016_1', 'Office_001_1']

    with multiprocessing.Pool(processes=8) as pool:
        args0 = scene_list
        pool.map(mp_run_wrapper, list(zip(args0)))
        pool.close()


if __name__ == "__main__":
    main()
