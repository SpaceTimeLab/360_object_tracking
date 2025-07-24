import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.config import get_cfg
import torch
from detectron2.data import DatasetCatalog
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import random
import cv2
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import json
from panoramic_detection import improved_OD
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from panoramic_detection.improved_OD import load_model
from utils.seed_helper import set_seed

# def load_model(model_type, min_size, max_size, score_threshold, nms_threshold, is_pano):
#     # first get the default config
#     cfg = get_cfg()
#
#     # choose a model from detectron2's model zoo
#     cfg.merge_from_file(
#         model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#     )
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#         "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#     )
#     if is_pano:
#         cfg.INPUT.MIN_SIZE_TEST = 0
#     else:
#         cfg.INPUT.MIN_SIZE_TEST = min_size  # set the size of the input images, if 0 then no resize
#     cfg.INPUT.MAX_SIZE_TEST = max_size
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
#         score_threshold  # set the threshold of the confidence score
#     )
#     # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
#     cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold  # set the NMS threshold
#
#     # set the device to use (GPU or CPU)
#     if torch.cuda.is_available():
#         print("GPU is available")
#         cfg.MODEL.DEVICE = "cuda"
#     else:
#         cfg.MODEL.DEVICE = "cpu"
#
#     # create a predictor instance with the config above
#     predictor_faster_rcnn = DefaultPredictor(cfg)
#     yolo_cfg = dict()
#     # TODO: need to change the name of yolo_cfg for fasterRCNN
#     if model_type == "Faster RCNN":
#         if min_size != 0:
#             yolo_cfg['imgsz'] = (min_size, 2 * min_size)
#         return predictor_faster_rcnn, cfg, yolo_cfg
#     elif model_type == "YOLO":
#         predictor_yolo = YOLO("yolo12n.pt")
#         # if torch.cuda.is_available():
#         #     predictor_yolo.to("cuda")
#         # min_size == 0 means we don't do resizing on the input image
#         if min_size != 0:
#             yolo_cfg['imgsz'] = (min_size, 2 * min_size)
#         yolo_cfg['conf'] = score_threshold
#         yolo_cfg['iou'] = nms_threshold
#         return predictor_yolo, cfg, yolo_cfg
#     return None

def compute_img_id_offsets(gt_paths):
    """
    Returns a list of cumulative image_id offsets, one per gt file.
    First offset is 0, second is max images in file1, third is sum of max images in file1+file2, etc.
    """
    offsets = []
    cum_offset = 0
    for p in gt_paths:
        offsets.append(cum_offset)
        data = json.load(open(p))
        max_id = max(img["id"] for img in data["images"])
        cum_offset += max_id
    return offsets

def merge_coco_gts_in_memory(gt_paths, img_id_offsets):
    """
    Build a merged COCO object entirely in memory, shifting each file’s images and annotations
    by the provided img_id_offsets.
    Returns:
      - coco_gt: merged COCO object
    """
    # Load first GT to get categories, then clear images/anns
    coco_gt = COCO(gt_paths[0])
    cats = coco_gt.dataset['categories']
    coco_gt.dataset['images'] = []
    coco_gt.dataset['annotations'] = []
    coco_gt.dataset['categories'] = cats

    ann_offset = 0
    # For each GT file + its offset:
    for offset, p in zip(img_id_offsets, gt_paths):
        data = json.load(open(p))

        # Shift & append images
        for img in data['images']:
            img['id'] += offset
            coco_gt.dataset['images'].append(img)

        # Shift & append annotations
        for ann in data['annotations']:
            ann['id']       += ann_offset
            ann['image_id'] += offset
            coco_gt.dataset['annotations'].append(ann)

        # Bump annotation offset to avoid collisions
        ann_offset = max(a['id'] for a in coco_gt.dataset['annotations']) + 1

    coco_gt.createIndex()
    return coco_gt

def merge_coco_dts_in_memory(dt_paths, img_id_offsets):
    """
    Concatenate all detection dicts in memory, shifting each det’s image_id
    by the corresponding offset.
    Returns the flat list of all detections.
    """
    all_dets = []
    for det_path, offset in zip(dt_paths, img_id_offsets):
        dets = json.load(open(det_path))
        for det in dets:
            det['image_id'] += offset
        all_dets.extend(dets)
    return all_dets

def main(args):
    SEED = 42
    set_seed(SEED)
    # make sure INFO (or DEBUG) logs go to your console
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("detectron2.engine.defaults").setLevel(logging.INFO)

    # 1. Define your “real” COCO IDs and your class names:
    COCO_IDS = [0, 1, 2, 3, 5, 7, 9]
    CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "bus", "truck", "traffic light"]

    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    FOV = 120
    THETAs = [0, 90, 180, 270]
    PHIs = [-10, -10, -10, -10]

    min_size = args.short_edge_size
    max_size = 10000
    score_threshold = args.score_threshold
    nms_threshold = args.nms_threshold

    model_type = args.model_type
    print("start evaluation on 3 videos: video1, video2, video3.")
    file_names = ["video1", "video2", "video3"]
    for file_name in file_names:

        # args.pano
        model, cfg, yolo_cfg = load_model(model_type, min_size, max_size, 5376 / 2688, score_threshold,
                                          nms_threshold)

        # 2. Register your dataset, telling Detectron2 how to map
        #    your JSON’s category_id (which you’ve numbered 1–7) back
        #    to the real COCO IDs above.
        def register_my_dataset(name, json_file, image_root):
            # build a map:  json_id (1–7)  →  contiguous id (0–6)

            dataset_id_to_contiguous = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
            MetadataCatalog.get(f"{file_name}_val").set(
                json_file=json_file,
                image_root=image_root,
                evaluator_type="coco",
                thing_classes=CLASS_NAMES,
                thing_dataset_id_to_contiguous_id=dataset_id_to_contiguous,
                thing_colors=[MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_colors[i] for i in COCO_IDS]
            )
            DatasetCatalog.register(
                name,
                lambda: load_coco_json(json_file, image_root, name),
            )

        register_my_dataset(
            f"{file_name}_val",
            f"./videos/{file_name}/COCO/annotations/instances_default.json",
            f"./videos/{file_name}/COCO/val"
        )

        # Tell Detectron2 how many classes and what their names are:
        MetadataCatalog.get("my_val").thing_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck',
                                                       'traffic light']
        # Now the metadata has been populated:
        print(MetadataCatalog.get("my_val").thing_classes)


        # build the sampler, make it 400 as the full set since the last 20 samples are not correctly labeled
        # test_set_sampler = InferenceSampler(400)

        # Build a DataLoader for the "my_val" split
        # TODO: the image size has problem.
        val_loader = build_detection_test_loader(cfg, f"{file_name}_val")

        # ---------- visualise the test dataset with bbox ----------
        # dataset_dicts = DatasetCatalog.get("my_val")
        # samples = random.sample(dataset_dicts, 3)
        # samples = dataset_dicts[:20]
        # metadata = MetadataCatalog.get("my_val")
        # for d in samples:
        #     img = cv2.imread(d["file_name"])  # BGR uint8
        #     vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        #     out = vis.draw_dataset_dict(d)  # draws d["annotations"]
        #     out_rgb = out.get_image()  # RGB H×W×3
        #
        #     plt.figure(figsize=(20, 12))
        #     plt.imshow(out_rgb)
        #     plt.axis("off")
        #     plt.show()
        results = []

        for batch in tqdm(val_loader):
            image = batch[0]["image"].permute(1, 2, 0).cpu().numpy() # (C, H, W)
            video_height, video_width = batch[0]["height"], batch[0]["width"]
            image_id = batch[0]["image_id"]

            # Your model’s output
            with torch.no_grad():
                bboxes, classes, scores = improved_OD.predict_one_frame(
                    FOV,
                    THETAs,
                    PHIs,
                    image,
                    model,
                    video_width,
                    video_height,
                    COCO_IDS,
                    True,
                    args.pano,
                    model_type,
                    False,
                    yolo_cfg
                )
            if not bboxes:
                continue
            bboxes = np.array(bboxes)
            bboxes = bboxes.astype(np.float64)

            if bboxes.ndim == 1:
                bboxes = bboxes[np.newaxis, :]

            scale_x = video_width / batch[0]["image"].shape[2]
            scale_y = video_height / batch[0]["image"].shape[1]
            # scale: x0, y0, x1, y1
            bboxes[:, [0, 2]] *= scale_x
            bboxes[:, [1, 3]] *= scale_y
            bboxes = bboxes.tolist()

            # --------- visualisation ---------
            # new_inst = Instances((video_height, video_width))
            #
            # new_inst.pred_boxes = Boxes(bboxes)
            # new_inst.pred_classes = torch.tensor(classes)
            # new_inst.scores = torch.tensor(scores)
            # im = cv2.imread(batch[0]['file_name'])
            #
            # v1 = Visualizer(
            #     im[:, :, ::-1],
            #     MetadataCatalog.get("my_val"),
            #     scale=1.0,
            # )
            # im2 = v1.draw_instance_predictions(new_inst)
            #
            # plt.figure(figsize=(20, 12))
            # plt.imshow(im2.get_image())
            # plt.axis("off")  # hide axes
            # plt.show()
            # --------- end of visualisation ---------

            # Parse the output: assuming your model returns boxes, labels, scores
            # boxes = outputs["boxes"].cpu().numpy()  # [N, 4]
            # scores = outputs["scores"].cpu().numpy()  # [N]
            # labels = outputs["labels"].cpu().numpy()  # [N]

            for bbox, score, one_class in zip(bboxes, scores, classes):
                x1, y1, x2, y2 = bbox
                result = {
                    "image_id": int(image_id),
                    "category_id": int(one_class + 1),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score)
                }
                results.append(result)

        # Save predictions to JSON
        with open(f"./videos/{file_name}/COCO/predictions.json", "w") as f:
            json.dump(results, f)



    coco_gts = [
        "./videos/video1/COCO/annotations/instances_default.json",
        "./videos/video2/COCO/annotations/instances_default.json",
        "./videos/video3/COCO/annotations/instances_default.json",
    ]
    coco_dts = [
        "./videos/video1/COCO/predictions.json",
        "./videos/video2/COCO/predictions.json",
        "./videos/video3/COCO/predictions.json",
    ]

    # 1) Compute image-ID offsets once
    img_id_offsets = compute_img_id_offsets(coco_gts)

    # 2) Merge GTs in memory with those offsets
    coco_gt = merge_coco_gts_in_memory(coco_gts, img_id_offsets)

    # 3) Merge DTs in memory with the same offsets
    all_dets = merge_coco_dts_in_memory(coco_dts, img_id_offsets)

    # 4) Load detections and run a single COCOeval
    coco_dt   = coco_gt.loadRes(all_dets)
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.params.areaRng = [
        [0**2,    1e5**2],   # all
        [0**2,    128**2],   # small
        [128**2,  384**2],   # medium
        [384**2,  1e5**2],   # large
    ]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    # for coco_gt, coco_dt in zip(coco_gts, coco_dts):
    #     print(f"\n=== Evaluating GT={coco_gt} vs DT={coco_dt} ===")
    #     coco_gt_obj = COCO(coco_gt)
    #     coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
    #     evaluator = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
    #     # our definition of small, medium and large.
    #     evaluator.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 128 ** 2], [128 ** 2, 384 ** 2], [384 ** 2, 1e5 ** 2]]
    #     evaluator.evaluate()
    #     evaluator.accumulate()
    #     evaluator.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Your program description here"
    )

    parser.add_argument(
        '-p', '--pano',
        action='store_true',
        help='Use our proposed method if set true'
    )
    parser.add_argument(
        '--sub_image_size',
        type=int,
        default=640,
        help='the size of sub image'
    )

    parser.add_argument(
        '-m', '--model_type',
        type=str,
        default="Faster RCNN",
        help='the object detection model'
    )

    parser.add_argument(
        '-s', '--short_edge_size',
        type=int,
        default=0,
        help='the length of short edge, default to 0 which means use the original image size'
    )

    parser.add_argument(
        '-n', '--nms_threshold',
        type=float,
        default=0.5,
        help='threshold for non-maxima suppression'
    )

    parser.add_argument(
        '-sc', '--score_threshold',
        type=float,
        default=0.5,
        help='threshold for non-maxima suppression'
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
