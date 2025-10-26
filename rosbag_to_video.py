#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ROS 2 imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge


IMAGE_TYPE = "sensor_msgs/msg/Image"
COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"


# --------- Utilities ---------
def is_ros2_bag_dir(p: Path) -> bool:
    """ROS 2 bag URIs are directories that contain metadata.yaml."""
    return p.is_dir() and (p / "metadata.yaml").exists()


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_bag_uris(source: Path) -> List[Path]:
    """
    If source is a single bag directory, return [source].
    If source is a directory containing multiple ROS 2 bag directories, return all of them,
    sorted naturally so *_0, *_1, *_2 … are processed in order.
    Also tolerate a path that points directly to a .db3 file by taking its parent.
    """
    if source.is_file() and source.suffix == ".db3":
        source = source.parent

    if is_ros2_bag_dir(source):
        return [source]

    if source.is_dir():
        candidates = [p for p in source.iterdir() if is_ros2_bag_dir(p)]
        if not candidates:
            raise FileNotFoundError(
                f"No ROS 2 bag directories with metadata.yaml found in: {source}"
            )
        return sorted(candidates, key=lambda p: natural_sort_key(p.name))

    raise FileNotFoundError(f"Path not found or not a ROS 2 bag: {source}")


def open_reader(uri: Path) -> SequentialReader:
    so = StorageOptions(uri=str(uri), storage_id="sqlite3")
    co = ConverterOptions(input_serialization_format="", output_serialization_format="")
    reader = SequentialReader()
    reader.open(so, co)
    return reader


def find_topic_type(reader: SequentialReader, topic_name: str) -> Optional[str]:
    for t in reader.get_all_topics_and_types():
        if t.name == topic_name:
            return t.type
    return None


def ensure_video_writer(
    path: str, width: int, height: int, fps: float = 20.0
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {path}")
    return vw


def decode_frame(msg_type: str, msg_obj, bridge: CvBridge) -> np.ndarray:
    if msg_type == IMAGE_TYPE:
        # desired BGR8 for OpenCV
        return bridge.imgmsg_to_cv2(msg_obj, desired_encoding="bgr8")
    elif msg_type == COMPRESSED_TYPE:
        # msg_obj.data is bytes
        buf = np.frombuffer(msg_obj.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode CompressedImage")
        return img
    else:
        raise ValueError(f"Unsupported message type for decoding: {msg_type}")


# --------- Core processing ---------
def process_bags(
    bag_uris: List[Path],
    topic: str,
    video_out: str,
    first_image_prefix: Optional[str],
    width: int,
    height: int,
    expected_type_hint: Optional[str] = None,
    fps: float = 20.0,
) -> int:
    """
    Iterate all bags in order, append frames to one video, save first frame of each bag.
    Returns total frame count written.
    """
    bridge = CvBridge()
    vw = ensure_video_writer(video_out, width, height, fps=fps)
    total_frames = 0

    for idx, uri in enumerate(bag_uris):
        print(f"\n[INFO] Processing bag: {uri}")
        reader = open_reader(uri)

        # Resolve topic type in this bag
        topic_type = find_topic_type(reader, topic)
        if topic_type is None:
            print(f"[WARN] Topic '{topic}' not found in {uri}, skipping this bag.")
            continue

        if expected_type_hint and expected_type_hint != topic_type:
            print(
                f"[WARN] Topic type mismatch in {uri}. Expected hint {expected_type_hint}, found {topic_type}"
            )

        # Prepare deserializer for this topic
        msg_cls = get_message(topic_type)

        first_image_saved = False

        # Iterate all messages and select only the desired topic
        while reader.has_next():
            topic_name, serialized, t = reader.read_next()
            if topic_name != topic:
                continue

            msg = deserialize_message(serialized, msg_cls)

            try:
                frame = decode_frame(topic_type, msg, bridge)
            except Exception as e:
                print(f"[WARN] Failed to decode a frame in {uri}: {e}")
                continue

            # Resize to output geometry if necessary
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_LINEAR
                )

            # Save the first image of this bag (if requested)
            if first_image_prefix and not first_image_saved:
                img_path = f"{first_image_prefix}_first_image_{idx}.png"
                cv2.imwrite(img_path, frame)
                print(f"[INFO] Saved first image: {img_path}")
                first_image_saved = True

            vw.write(frame)
            total_frames += 1

    vw.release()
    return total_frames


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(
        description="ROS 2: Convert one or many rosbag2 (sqlite3) bags to a single MP4."
    )
    parser.add_argument(
        "-source",
        dest="source",
        required=True,
        help="Path to a ROS 2 bag directory (with metadata.yaml) OR a directory containing multiple such bags OR a .db3 file.",
    )
    parser.add_argument(
        "-topic",
        dest="topic",
        required=True,
        help="Image topic name to extract, e.g. /camera/image_raw or /camera/image_raw/compressed",
    )
    parser.add_argument(
        "-output",
        dest="video_path",
        required=True,
        help="Output MP4 path",
    )
    parser.add_argument(
        "-first_image",
        dest="first_image_prefix",
        default=None,
        help="If set, saves the first frame of each bag as <prefix>_first_image_<idx>.png",
    )
    parser.add_argument(
        "-width", dest="width", type=int, default=1280, help="Output frame width"
    )
    parser.add_argument(
        "-height", dest="height", type=int, default=720, help="Output frame height"
    )
    parser.add_argument(
        "-fps", dest="fps", type=float, default=30.0, help="Output video FPS"
    )
    parser.add_argument(
        "-type_hint",
        dest="type_hint",
        choices=[IMAGE_TYPE, COMPRESSED_TYPE],
        default=None,
        help="Optional expected type to sanity-check deserialization.",
    )

    args = parser.parse_args()

    src = Path(args.source).resolve()
    try:
        bag_uris = list_bag_uris(src)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("[INFO] Found bags in order:")
    for i, u in enumerate(bag_uris):
        print(f"  {i:02d}: {u}")

    total = process_bags(
        bag_uris=bag_uris,
        topic=args.topic,
        video_out=args.video_path,
        first_image_prefix=args.first_image_prefix,
        width=args.width,
        height=args.height,
        expected_type_hint=args.type_hint,
        fps=args.fps,
    )

    print(f"\n[OK] Wrote {total} frames to {args.video_path}")


if __name__ == "__main__":
    main()