import sys, os
# # assume script lives alongside detectron2/
proj_root = os.path.dirname(__file__)
if proj_root not in sys.path:
    sys.path.insert(0, os.path.join(proj_root, "detectron2"))
import detectron2
print(detectron2.__file__)
from Overtaking_Detection import Overtaking_Detection
import argparse
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_dir", required=True, type=str)
    # parser.add_argument("--output_video_dir", required=True, type=str)
    parser.add_argument(
        "--mode", type=str, choices=["Confirmed", "Unconfirmed"], default="Unconfirmed"
    )
    parser.add_argument(
        "--prevent_different_classes_match", default=True, type=boolean_string
    )
    parser.add_argument("--match_across_boundary", default=True, type=boolean_string)
    parser.add_argument(
        "--classes_to_detect", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 9]
    )
    parser.add_argument(
        "--classes_to_detect_movement", nargs="+", type=int, default=[2, 5, 7]
    )

    parser.add_argument(
        "--size_thresholds",
        nargs="+",
        type=int,
        default=[400 * 400, 400 * 400, 400 * 400],
    )
    parser.add_argument("--FOV", type=int, default=120)
    parser.add_argument("--THETAs", nargs="+", type=int, default=[0, 90, 180, 270])
    parser.add_argument("--PHIs", nargs="+", type=int, default=[-10, -10, -10, -10])
    parser.add_argument("--sub_image_width", type=int, default=1280)
    parser.add_argument(
        "--model_type", type=str, choices=["YOLO", "Faster RCNN"], default="YOLO"
    )
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--use_mymodel", default=True, type=boolean_string)
    opt = parser.parse_args()
    print(opt)
    return opt


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def main(opt):
    video_dir = opt.input_video_dir
    videos = list_video_files(video_dir)
    print(f"Found {len(videos)} video(s):")
    for input_video_path in videos:
        print(input_video_path)
        output_video_path = input_video_path.with_name(input_video_path.stem + "_pred" + input_video_path.suffix)

        Overtaking_Detection(
            str(input_video_path),
            str(output_video_path),
            opt.mode,
            opt.prevent_different_classes_match,
            opt.match_across_boundary,
            opt.classes_to_detect,
            opt.classes_to_detect_movement,
            opt.size_thresholds,
            opt.FOV,
            opt.THETAs,
            opt.PHIs,
            opt.sub_image_width,
            opt.model_type,
            opt.score_threshold,
            opt.nms_threshold,
            opt.use_mymodel,
        )


def list_video_files(input_dir, extensions=None):
    """
    List all video files under input_dir.

    Args:
        input_dir (str or Path): Path to the directory to search.
        extensions (set of str): Video file extensions to include (without the dot).
                                 Defaults to common video formats.
        recursive (bool): If True, search subdirectories.

    Returns:
        List[Path]: Sorted list of Paths to video files.
    """
    if extensions is None:
        extensions = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'mpeg', 'mpg'}

    input_dir = Path(input_dir)
    video_files = []

    for path in input_dir.iterdir():
        if path.is_file() and path.suffix.lower().lstrip('.') in extensions:
            video_files.append(path)

    return sorted(video_files)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
