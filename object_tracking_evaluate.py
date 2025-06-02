import argparse

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from deep_sort.deep_sort import DeepSort
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from panoramic_detection import improved_OD as OD


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
    predictor_faster_rcnn = DefaultPredictor(cfg)
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


# function used to realize object tracking on a panoramic video
def Object_Tracking(
        input_video_path,
        MOT_text_path,
        prevent_different_classes_match=True,
        match_across_boundary=True,
        classes_to_detect=[0, 1, 2, 3, 5, 7, 9],
        FOV=120,
        THETAs=[0, 90, 180, 270],
        PHIs=[-10, -10, -10, -10],
        sub_image_width=640,
        model_type="YOLO",
        score_threshold=0.4,
        nms_threshold=0.45,
        use_mymodel=True,
        min_size = 0,
        max_size = 10000,
):
    print("Loading Model...")

    model, cfg, yolo_cfg = load_model(model_type, min_size, max_size, score_threshold, nms_threshold)

    # # load the pretrained detection model
    # model, cfg = OD.load_model(
    #     model_type, sub_image_width, score_threshold, nms_threshold
    # )

    print("Model Loaded!")

    # read the input panoramic video (of equirectangular projection)
    video_capture = cv2.VideoCapture(input_video_path)

    # if the input path is not right, warn the user
    if not video_capture.isOpened():
        print("Can not open the video file.")
        exit(0)
    # if right, get some info about the video (width, height, frame count and fps)
    else:
        video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # output the video info
        print(
            "The input video is "
            + str(video_width)
            + " in width and "
            + str(video_height)
            + " in height."
        )


    # create a deepsort instance with the pre-trained feature extraction model
    deepsort = DeepSort(
        "./deep_sort/deep/checkpoint/ckpt.t7", use_cuda=torch.cuda.is_available()
    )

    # the number of current frame
    num_of_frame = 1

    with open(MOT_text_path, "w") as f:
        # for each image frame in the video
        while video_capture.grab():

            # get the next image frame
            _, im = video_capture.retrieve()

            # get the predictions on the current frame
            # TODO: currently only works for YoLo detection
            bboxes, classes_all, scores_all = OD.predict_one_frame(
                FOV,
                THETAs,
                PHIs,
                im,
                model,
                video_width,
                video_height,
                sub_image_width,
                classes_to_detect,
                False,
                use_mymodel,
                model_type,
                not match_across_boundary,
                yolo_cfg
            )

            # TODO: Resize the boxes, seems no need for YoLo


            # convert the bboxes from [x,y,x,y] to [xc,yc,w,h]
            bboxes_all_xcycwh = OD.xyxy2xcycwh(bboxes)

            # update deepsort and get the tracking results
            track_outputs = deepsort.update(
                np.array(bboxes_all_xcycwh),
                np.array(classes_all),
                np.array(scores_all),
                im,
                prevent_different_classes_match,
                match_across_boundary,
            )

            # plot the results on the video and save them as MOT texts
            if len(track_outputs) > 0:
                bbox_xyxy = track_outputs[:, :4]
                track_classes = track_outputs[:, 4]
                track_scores = track_outputs[:, 5]
                identities = track_outputs[:, -1]

                for bb_xyxy, track_class, track_score, identity in zip(
                        bbox_xyxy, track_classes, track_scores, identities
                ):
                    f.write(
                        str(num_of_frame)
                        + ","
                        + str(int(identity))
                        + ","
                        + str(deepsort._xyxy_to_tlwh(bb_xyxy))
                        .strip("(")
                        .strip(")")
                        .replace(" ", "")
                        + ","
                        + str(track_score)
                        + ","
                        + str(track_class)
                        + ",-1,-1\n"
                    )

            num_of_frame += 1

    # release the input and output videos
    video_capture.release()

    print("Output Finished!")

    # Load GT and predictions
    gt = read_mot_file("/Users/supernova/360_object_tracking/video1/MOT/gt/gt_id_merged.txt")  # ground truth
    pred = read_mot_file(MOT_text_path)  # your DeepSORT output

    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Group by frame
    for frame in sorted(gt['frame'].unique()):
        gt_frame = gt[gt['frame'] == frame]
        pred_frame = pred[pred['frame'] == frame]

        gt_boxes = list(zip(gt_frame['x'], gt_frame['y'], gt_frame['right'], gt_frame['bottom']))
        pred_boxes = list(zip(pred_frame['x'], pred_frame['y'], pred_frame['right'], pred_frame['bottom']))

        # max_iou=0.5 is the default value.
        distance = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

        acc.update(gt_frame['id'].values, pred_frame['id'].values, distance)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='DeepSORT')

    # Print results
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


def read_mot_file(filepath):
    df = pd.read_csv(filepath, header=None, usecols=range(6))
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
    df['right'] = df['x'] + df['w']
    df['bottom'] = df['y'] + df['h']
    return df


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def main(opt):
    Object_Tracking(
        opt.input_video_path,
        opt.output_video_path,
        opt.MOT_text_path,
        opt.prevent_different_classes_match,
        opt.match_across_boundary,
        opt.classes_to_detect,
        opt.FOV,
        opt.THETAs,
        opt.PHIs,
        opt.sub_image_width,
        opt.model_type,
        opt.score_threshold,
        opt.nms_threshold,
        opt.use_mymodel,
        opt.short_edge_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--input_video_path", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    parser.add_argument("--MOT_text_path", required=True, type=str)
    parser.add_argument(
        "--prevent_different_classes_match", default=True, type=boolean_string
    )
    parser.add_argument("--match_across_boundary", default=True, type=boolean_string)
    parser.add_argument(
        "--classes_to_detect", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 9]
    )
    parser.add_argument("--FOV", type=int, default=120)
    parser.add_argument("--THETAs", nargs="+", type=int, default=[0, 90, 180, 270])
    parser.add_argument("--PHIs", nargs="+", type=int, default=[-10, -10, -10, -10])
    parser.add_argument("--sub_image_width", type=int, default=640)
    parser.add_argument("--short_edge_size", type=int, default=0)
    parser.add_argument(
        "--model_type", type=str, choices=["YOLO", "Faster RCNN"], default="YOLO"
    )
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--use_mymodel", default=True, type=boolean_string)
    parsed_args = parser.parse_args()
    main(parsed_args)
