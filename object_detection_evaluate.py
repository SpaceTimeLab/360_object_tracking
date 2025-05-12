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
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import random
import cv2
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
import argparse
from ultralytics import YOLO

def load_model(model_type, min_size, max_size, score_threshold, nms_threshold):
    # first get the default config
    cfg = get_cfg()

    # choose a model from detectron2's model zoo
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.INPUT.MIN_SIZE_TEST = min_size  # set the size of the input images, if 0 then no resize
    cfg.INPUT.MAX_SIZE_TEST = max_size
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        score_threshold  # set the threshold of the confidence score
    )
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold  # set the NMS threshold

    # set the device to use (GPU or CPU)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

    # create a predictor instance with the config above
    predictor_faster_rcnn = DefaultPredictor(cfg).model
    if model_type == "Faster RCNN":
        return predictor_faster_rcnn, cfg, None
    elif model_type == "YOLO":
        predictor_yolo = YOLO("yolo12n.pt")
        yolo_cfg = dict()
        # min_size == 0 means we don't do resizing on the input image
        if min_size != 0:
            yolo_cfg['imgsz'] = (min_size, 2 * min_size)
        yolo_cfg['conf'] = score_threshold
        yolo_cfg['iou'] = nms_threshold
        return predictor_yolo, cfg, yolo_cfg
    return None


def main(args):

    # make sure INFO (or DEBUG) logs go to your console
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("detectron2.engine.defaults").setLevel(logging.INFO)

    # 1. Define your “real” COCO IDs and your class names:
    COCO_IDS = [0, 1, 2, 3, 5, 7, 9]
    CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "bus", "truck", "traffic light"]

    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.

    min_size = args.short_edge_size
    max_size = 10000
    score_threshold = 0.6
    nms_threshold=args.nms_threshold

    model_typle = args.model_type

    model, cfg, yolo_cfg = load_model(model_typle, min_size, max_size, score_threshold, nms_threshold)


    # 2. Register your dataset, telling Detectron2 how to map
    #    your JSON’s category_id (which you’ve numbered 1–7) back
    #    to the real COCO IDs above.
    def register_my_dataset(name, json_file, image_root):
        # build a map:  json_id (1–7)  →  contiguous id (0–6)

        dataset_id_to_contiguous = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        MetadataCatalog.get("my_val").set(
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
        "my_val",
        "/Users/supernova/360_object_tracking/video1/COCO/annotations/instances_default.json",
        "/Users/supernova/360_object_tracking/video1/COCO/val"
    )

    # Tell Detectron2 how many classes and what their names are:
    MetadataCatalog.get("my_val").thing_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck',
                                                   'traffic light']
    # Now the metadata has been populated:
    print(MetadataCatalog.get("my_val").thing_classes)


    # build the sampler
    test_set_sampler = InferenceSampler(400)

    # Build a DataLoader for the "my_val" split
    val_loader = build_detection_test_loader(cfg, "my_val", sampler=test_set_sampler)

    # ---------- visualise the test dataset with bbox ----------
    # 1) load the raw dicts
    dataset_dicts = DatasetCatalog.get("my_val")
    # 2) sample 3 at random
    # samples = random.sample(dataset_dicts, 3)
    # samples = dataset_dicts[-20:]
    # 3) visualize each
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

    # You can specify which tasks to compute; by default it infers from dataset (bbox, segm, keypoints…)
    evaluator = COCOEvaluator(
        dataset_name="my_val",  # the name you registered
        tasks=("bbox",),  # e.g., "bbox", "segm", or ("bbox", "segm")
        distributed=False,  # set True if using multi-GPU
        output_dir="./output"  # where to dump JSON results & summaries
    )
    # AssertionError: A prediction has class=11, but the dataset only has 7 classes and predicted class id should be in [0, 6].
    # We need to filter out unnecessary classes

    metrics = inference_on_dataset(
        model,  # or Trainer.model
        val_loader,
        evaluator,
        COCO_IDS,
        args.pano,
        cfg,
        model_typle,
        yolo_cfg,
        args.sub_image_size
    )
    print(metrics)


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

    parsed_args = parser.parse_args()

    main(parsed_args)
