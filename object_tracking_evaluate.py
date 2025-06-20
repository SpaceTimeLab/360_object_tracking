import argparse
import os

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from deep_sort.deep_sort import DeepSort
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from panoramic_detection import improved_OD as OD
from panoramic_detection.improved_OD import load_model
from strong_sort_new import StrongSort

def count_subfolders(root_path):
    """
    Count the number of immediate subfolders under root_path.
    """
    count = 0
    for entry in os.scandir(root_path):
        if entry.is_dir():
            count += 1
    return count

# function used to realize object tracking on a panoramic video
def Object_Tracking(
        input_video_dir,
        MOT_text_path,
        prevent_different_classes_match=True,
        match_across_boundary=True,
        classes_to_detect=[0, 1, 2, 3, 5, 7, 9],
        FOV=120,
        THETAs=[0, 90, 180, 270],
        PHIs=[-10, -10, -10, -10],
        # sub_image_width=640,
        detector="YOLO",
        score_threshold=0.4,
        nms_threshold=0.45,
        use_mymodel=True,
        min_size = 640, # min_size will be used as width for resizing
        max_size = 10000,
        tracker_name = "strongsort"
):
    if tracker_name == "deepsort":
        # create a tracker with the pre-trained feature extraction model
        tracker = DeepSort(
            "./deep_sort/deep/checkpoint/ckpt.t7", use_cuda=torch.cuda.is_available()
        )
    elif tracker_name == "strongsort":
        tracker = StrongSort(
            "./deep_sort/deep/checkpoint/ckpt.t7", use_cuda=torch.cuda.is_available()
        )

    num_of_videos = count_subfolders(input_video_dir)
    print(f"Number of videos: {num_of_videos}")
    # Create accumulator
    accs = []
    video_names = []
    for idx in range(1, num_of_videos + 1):
        if idx == 1:
            input_video_path = os.path.join(input_video_dir, f"video{idx}", f"video{idx}.mp4")
        else:
            input_video_path = os.path.join(input_video_dir, f"video{idx}", f"video{idx}.mov")
        # read the input panoramic video (of equirectangular projection)
        video_capture = cv2.VideoCapture(input_video_path)
        video_names.append(f"video{idx}")
        # if the input path is not right, warn the user
        if not video_capture.isOpened():
            print("Can not open the video file.")
            exit(0)
        # if right, get some info about the video (width, height, frame count and fps)
        else:
            print("Processing video: ", input_video_path, " ...")
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
            video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Loading Model...")
        model, cfg, yolo_cfg = load_model(detector, min_size, max_size, video_width / video_height, score_threshold, nms_threshold)
        print("Model Loaded!")

        # the number of current frame
        frame_idx = 1
        MOT_path = os.path.join(input_video_dir, f"video{idx}", "MOT")
        predicted_MOT_path = os.path.join(MOT_path, "predicted.txt")
        with open(predicted_MOT_path, "w") as f:
            pbar = tqdm(total=video_frame_count, desc="Processing frames")

            # for each image frame in the video
            while video_capture.grab() and frame_idx <= 400:

                # get the next image frame
                _, im = video_capture.retrieve()

                # get the predictions on the current frame
                # TODO: currently only works for YOLO detection
                bboxes, classes_all, scores_all = OD.predict_one_frame(
                    FOV,
                    THETAs,
                    PHIs,
                    im,
                    model,
                    video_width,
                    video_height,
                    classes_to_detect,
                    True,
                    use_mymodel,
                    detector,
                    match_across_boundary,
                    yolo_cfg
                )
                # convert the bboxes from [x,y,x,y] to [xc,yc,w,h]
                bboxes_all_xcycwh = OD.xyxy2xcycwh(bboxes)

                # update deepsort and get the tracking results
                track_outputs = tracker.update(
                    np.array(bboxes_all_xcycwh),
                    np.array(classes_all),
                    np.array(scores_all),
                    im,
                    prevent_different_classes_match,
                    match_across_boundary,
                )

                # save results as MOT texts
                if len(track_outputs) > 0:
                    bbox_xyxy = track_outputs[:, :4]
                    track_classes = track_outputs[:, 4]
                    track_scores = track_outputs[:, 5]
                    identities = track_outputs[:, -1]

                    for bb_xyxy, track_class, track_score, identity in zip(
                            bbox_xyxy, track_classes, track_scores, identities
                    ):
                        f.write(
                            str(frame_idx)
                            + ","
                            + str(int(identity))
                            + ","
                            + str(tracker._xyxy_to_tlwh(bb_xyxy))
                            .strip("(")
                            .strip(")")
                            .replace(" ", "")
                            + ","
                            + str(track_score)
                            + ","
                            + str(track_class + 1)
                            + ",-1,-1\n"
                        )
                frame_idx += 1
                pbar.update(1)

        # release the input and output videos
        video_capture.release()
        pbar.close()

        print("Output Finished!")

        # calculate metrics for current video

        acc = mm.MOTAccumulator(auto_id=True)
        # Load GT and predictions
        gt_path = os.path.join(MOT_path, "gt", "gt_id_merged.txt")
        gt = read_mot_file(gt_path)  # ground truth
        pred = read_mot_file(predicted_MOT_path)  # your DeepSORT output

        # Group by frame
        for frame in sorted(gt['frame'].unique()):
            gt_frame = gt[gt['frame'] == frame]
            pred_frame = pred[pred['frame'] == frame]
            # TODO: find out why there is no pred_boxes in the first and second frame
            gt_boxes = list(zip(gt_frame['x'], gt_frame['y'], gt_frame['w'], gt_frame['h']))
            pred_boxes = list(zip(pred_frame['x'], pred_frame['y'], pred_frame['w'], pred_frame['h']))

            # max_iou=0.5 is the default value.
            distance = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

            acc.update(gt_frame['id'].values, pred_frame['id'].values, distance)
        accs.append(acc)
    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=video_names, # given we only have 3 videos for evaluation
        generate_overall=True
    )
    # Print results
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

# TODO: maybe wrong. ...The MOT file doesn't follow the standard bbox format, so we need to convert it for evaluation metrics
def read_mot_file(filepath):
    df = pd.read_csv(filepath, header=None, usecols=range(6))
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
    # df['right'] = df['x'] + df['w']
    # df['bottom'] = df['y'] + df['h']
    return df


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def main(opt):
    Object_Tracking(
        opt.input_video_path,
        opt.MOT_text_path,
        opt.prevent_different_classes_match,
        opt.match_across_boundary,
        opt.classes_to_detect,
        opt.FOV,
        opt.THETAs,
        opt.PHIs,
        opt.detector,
        opt.score_threshold,
        opt.nms_threshold,
        opt.use_mymodel,
        opt.short_edge_size,
        opt.tracker
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
    # parser.add_argument("--sub_image_width", type=int, default=640)
    parser.add_argument("--short_edge_size", type=int, default=640)
    parser.add_argument(
        "--detector", type=str, choices=["YOLO", "Faster RCNN"], default="YOLO"
    )
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--use_mymodel", default=True, type=boolean_string)
    parser.add_argument(
        "--tracker", type=str, choices=["deepsort", "strongsort"], default="deepsort"
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
