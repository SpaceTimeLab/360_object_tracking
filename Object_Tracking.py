import argparse
import time
import torch
import cv2
import numpy as np
from torchvision.ops import batched_nms
from ultralytics import YOLO

from panoramic_detection import improved_OD as OD
from deep_sort.deep_sort import DeepSort
from panoramic_detection.draw_output import draw_boxes

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from panoramic_detection.improved_OD import load_model
from strong_sort_new import StrongSort


# function used to realize object tracking on a panoramic video
def Object_Tracking(
    input_video_path,
    output_video_path,
    # MOT_text_path,
    prevent_different_classes_match=True,
    match_across_boundary=True,
    classes_to_detect=[0, 1, 2, 3, 5, 7, 9],
    FOV=120,
    THETAs=[0, 90, 180, 270],
    PHIs=[-10, -10, -10, -10],
    # sub_image_width=640,
    model_type="YOLO",
    score_threshold=0.4,
    nms_threshold=0.45,
    use_mymodel=True,
    min_size=640,  # min_size will be used as width for resizing
    max_size=10000,
):

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
        video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        outputfile = cv2.VideoWriter(
            output_video_path, fourcc, video_fps, (video_width, video_height)
        )

        # output the video info
        print(
            "The input video is "
            + str(video_width)
            + " in width and "
            + str(video_height)
            + " in height."
        )

    print("Loading Model...")

    model, cfg, yolo_cfg = load_model(model_type, min_size, max_size, video_width / video_height, score_threshold, nms_threshold)

    # # load the pretrained detection model
    # model, cfg = OD.load_model(
    #     model_type, sub_image_width, score_threshold, nms_threshold
    # )

    print("Model Loaded!")

    # create a deepsort instance with the pre-trained feature extraction model
    tracker = StrongSort(
        "./deep_sort/deep/checkpoint/ckpt.t7", use_cuda=torch.cuda.is_available()
    )

    # the number of current frame
    num_of_frame = 1

    # for each image frame in the video
    while video_capture.grab():
        time1 = time.time()
        # get the next image frame
        _, im = video_capture.retrieve()
        # get the predictions on the current frame
        bboxes_all, classes_all, scores_all = OD.predict_one_frame(
            FOV,
            THETAs,
            PHIs,
            im,
            model,
            video_width,
            video_height,
            # sub_image_width,
            classes_to_detect,
            True,
            use_mymodel,
            model_type,
            not match_across_boundary, # False means do not split image2
            yolo_cfg
        )
        # convert the bboxes from [x,y,x,y] to [xc,yc,w,h]
        bboxes_all_xcycwh = OD.xyxy2xcycwh(bboxes_all)
        # update deepsort and get the tracking results
        track_outputs = tracker.update(
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
            im = draw_boxes(
                im, bbox_xyxy, track_classes, track_scores, video_width, identities
            )
        outputfile.write(im)
        # show the current FPS
        time2 = time.time()
        if num_of_frame % 5 == 0:
            print(num_of_frame, "/", video_frame_count)
            print(str(1 / (time2 - time1)) + " fps")
        num_of_frame += 1

    # release the input and output videos
    video_capture.release()
    outputfile.release()

    print("Output Finished!")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_path", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    # parser.add_argument("--MOT_text_path", required=True, type=str)
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
    parser.add_argument("--short_edge_size", type=int, default=1280)
    parser.add_argument(
        "--model_type", type=str, choices=["YOLO", "Faster RCNN"], default="YOLO"
    )
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--use_mymodel", default=True, type=boolean_string)
    opt = parser.parse_args()
    # print(opt)
    return opt


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def main(opt):
    Object_Tracking(
        opt.input_video_path,
        opt.output_video_path,
        # opt.MOT_text_path,
        opt.prevent_different_classes_match,
        opt.match_across_boundary,
        opt.classes_to_detect,
        opt.FOV,
        opt.THETAs,
        opt.PHIs,
        opt.model_type,
        opt.score_threshold,
        opt.nms_threshold,
        opt.use_mymodel,
        opt.short_edge_size
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
