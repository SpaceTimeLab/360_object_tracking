import cv2
import os
import glob
import json
from PIL import Image, ImageDraw
from typing import List, Tuple

# ------------------------------------------------------------------------------
# extract_frames
# ------------------------------------------------------------------------------
def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = None,
    resize: Tuple[int,int] = None
) -> int:
    """
    Read a video file and dump individual frames as PNGs.

    Args:
        video_path:  Path to the input .mp4 video.
        output_dir:  Directory where extracted frames will be saved.
        target_fps:  If provided, evenly sample frames to match this FPS;
                     if None, extract every frame.
        resize:      (width, height) tuple; if provided, each frame is
                     resized before saving.

    Returns:
        Number of frames written to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Verify video exists and can be opened
    if not os.path.isfile(video_path):
        print(f"[WARN] Video not found: {video_path}")
        return 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Couldn't open video: {video_path}")
        return 0

    # 2) Compute interval for sampling frames
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps or 1
    interval = 1 if target_fps is None else max(1, int(src_fps / target_fps))

    # 3) Loop through frames and save every `interval`-th frame
    count = 0
    out_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video
        if count % interval == 0:
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            filename = f"frame_{out_idx:06d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            out_idx += 1
        count += 1

    cap.release()
    print(f"[INFO] Extracted {out_idx} frames to {output_dir}")
    return out_idx


# ------------------------------------------------------------------------------
# find_video_file
# ------------------------------------------------------------------------------
def find_video_file(folder: str, preferred: str) -> str:
    """
    Locate the MP4 in a folder.

    1) Try `folder/preferred`.
    2) If not found, fall back to the first .mp4 in `folder`.

    Returns:
        Full path to the chosen .mp4, or empty string if none found.
    """
    preferred_path = os.path.join(folder, preferred)
    if os.path.isfile(preferred_path):
        return preferred_path

    # Fallback: pick any .mp4 present
    candidates = glob.glob(os.path.join(folder, "*.mp4"))
    if candidates:
        print(f"[INFO] Using {os.path.basename(candidates[0])} instead of missing {preferred}")
        return candidates[0]

    return ""


# ------------------------------------------------------------------------------
# load_coco_annotations
# ------------------------------------------------------------------------------
def load_coco_annotations(json_path: str):
    """
    Parse COCO-format JSON file into lookup tables.

    - imgs:  image_id → image metadata dict
    - annos: image_id → list of annotation dicts
    - cats:  category_id → category name

    Returns empty dicts if JSON not found.
    """
    if not os.path.isfile(json_path):
        print(f"[WARN] COCO JSON not found: {json_path}")
        return {}, {}, {}

    coco = json.load(open(json_path, 'r'))
    imgs = {img['id']: img for img in coco.get('images', [])}

    annos = {}
    for ann in coco.get('annotations', []):
        annos.setdefault(ann['image_id'], []).append(ann)

    cats = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    return imgs, annos, cats


# ------------------------------------------------------------------------------
# visualise_annotations
# ------------------------------------------------------------------------------
def visualise_annotations(
    image_dir: str,
    json_path: str,
    output_dir: str,
    sample_ids: List[int] = None
):
    """
    Overlay COCO bounding boxes and labels onto extracted frames.

    Args:
        image_dir:   Directory containing the frames (PNG files).
        json_path:   Path to COCO JSON file (annotations folder).
        output_dir:  Directory to save the visualised images.
        sample_ids:  Optional list of image IDs to visualise; defaults to all.
    """
    os.makedirs(output_dir, exist_ok=True)
    imgs, annos, cats = load_coco_annotations(json_path)
    if not imgs:
        return

    # Iterate over each image ID to draw its annotations
    for img_id in (sample_ids or imgs.keys()):
        info = imgs.get(img_id)
        if not info:
            continue

        img_file = os.path.join(image_dir, info['file_name'])
        if not os.path.isfile(img_file):
            print(f"[WARN] Image not found: {img_file}")
            continue

        img = Image.open(img_file).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw each bbox and label
        for ann in annos.get(img_id, []):
            x, y, w, h = ann['bbox']
            label = cats.get(ann['category_id'], str(ann['category_id']))
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            draw.text((x, y - 10), label, fill="red")

        out_name = f"vis_{info['file_name']}"
        img.save(os.path.join(output_dir, out_name))

    print(f"[INFO] Saved visualisations to {output_dir}")


# ------------------------------------------------------------------------------
# process_video
# ------------------------------------------------------------------------------
def process_video(
    base_dir: str,
    video_fname: str,
    target_fps: float = 30,
    resize: Tuple[int,int] = None
):
    """
    Full pipeline for one video folder:
      1) Locate the video file
      2) Extract frames to <base_dir>/val
      3) Visualise annotations to <base_dir>/COCO/val_visualise
    """
    # 1) Find the video .mp4
    video_path = find_video_file(base_dir, video_fname)
    if not video_path:
        print(f"[WARN] No MP4 found in {base_dir}")
        return

    # 2) Extract frames
    frame_dir = os.path.join(base_dir, "val")
    n = extract_frames(video_path, frame_dir, target_fps, resize)
    if n == 0:
        return

    # 3) visualisation (IDs in COCO/annotations)
    json_after = os.path.join(base_dir, "COCO", "annotations", "instances_default.json")
    vis_after = os.path.join(base_dir, "COCO", "val_visualise")
    visualise_annotations(frame_dir, json_after, vis_after)


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
def main():
    """
    Entrypoint: for each of video1, video2, video3 under data_from_Jingwei/,
    run the full process_video() pipeline.
    """
    root = "./data_from_Jingwei"
    for v in ["video1", "video2", "video3"]:
        base = os.path.join(root, v)
        print(f"\n=== Processing {base} ===")
        process_video(
            base_dir=base,
            video_fname=f"{v}.mp4",
            target_fps=30,
            resize=(5376, 2688)
        )


if __name__ == "__main__":
    main()
