#!/usr/bin/env python3
"""
ROS 2 node: live TalkNCE active-speaker detection.

Subscribes:
  <image_topic>   (sensor_msgs/CompressedImage)  – compressed video from a camera topic

Publishes:
  <output_topic>  (sensor_msgs/CompressedImage)    – annotated frames with ASD bounding boxes
                                         (green = speaking, blue = silent, grey = pending)

Audio is captured from the default system microphone via sounddevice.

Usage
  python ros2_infer_live.py \\
    --checkpoint /path/to/model.model \\
    --image-topic /camera/image_raw \\
    --output-topic /talknce/image_annotated \\
    --fps 25 --buffer-seconds 4

Requirements:
  ros2, rclpy, cv_bridge, sounddevice, torch (CUDA), opencv-python
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
import mediapipe as mp

import cv2
import numpy as np
import resampy
import sounddevice as sd
import torch

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dlhammer.dlhammer import bootstrap
from loconet import loconet
from torchvggish import vggish_input


# ──────────────────────────── local defaults ────────────────────────────────

# Edit these defaults for your regular deployment setup so the CLI can stay
# short. Command-line flags still override any of these values when needed.
DEFAULT_CFG_PATH = "configs/test.yaml"
DEFAULT_IMAGE_TOPIC = "/zed/zed_node/rgb_raw/image_raw_color/compressed"
DEFAULT_OUTPUT_TOPIC = "/talknce/image_annotated"
DEFAULT_SPEAKER_BBOX_TOPIC = "/talknce/active_speaker_bbox"
DEFAULT_ALL_DETECTIONS_TOPIC = "/talknce/all_detections"
DEFAULT_EXPECTED_FPS = 25.0
DEFAULT_FRAME_TIME_SOURCE = "header"

DEFAULT_AUDIO_BACKEND = "parec"
# No hardcoded device: empty means parec/sounddevice fall back to the
# system's default input. Set TALKNCE_AUDIO_DEVICE (or pass --audio-device)
# for a specific mic, e.g. a PulseAudio source name from --list-pulse-sources.
DEFAULT_AUDIO_DEVICE = os.environ.get("TALKNCE_AUDIO_DEVICE") or None
DEFAULT_PULSE_SAMPLE_RATE = 48000
DEFAULT_PULSE_CHANNELS = 1
DEFAULT_PULSE_CHUNK_FRAMES = 1024

DEFAULT_DETECT_EVERY_N = 2
DEFAULT_BUFFER_SECONDS = 2.0
DEFAULT_INFERENCE_STRIDE = 2
DEFAULT_MIN_TRACK_FRAMES = 5
DEFAULT_MAX_TRACK_MISS = 10
DEFAULT_SCORE_THRESH = 0.3
DEFAULT_SCORE_EMA_ALPHA = 0.35
DEFAULT_SPEAK_OFF_DELTA = 0.15
DEFAULT_AUDIO_OFFSET_MS = 0.0
DEFAULT_AUDIO_OFFSET_SEARCH_MS = 0.0
DEFAULT_AUDIO_OFFSET_STEP_MS = 150.0


# ──────────────────────────── geometry helpers ────────────────────────────────

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(1e-6, area_a + area_b - inter))


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


# def detect_faces(gray, detector):
#     faces = detector.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32),
#         flags=cv2.CASCADE_SCALE_IMAGE,
#     )
#     if not isinstance(faces, np.ndarray) or len(faces) == 0:
#         return []
#     return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]

### mediapipe
def create_mediapipe_face_detector(model_path):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


def detect_faces_mediapipe(frame_bgr, detector, timestamp_ms):
    """
    Returns face boxes in (x1, y1, x2, y2) pixel coords.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    h, w = frame_bgr.shape[:2]
    boxes = []

    if result.detections:
        for det in result.detections:
            bbox = det.bounding_box
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(bbox.origin_x + bbox.width)
            y2 = int(bbox.origin_y + bbox.height)
            boxes.append(clip_box((x1, y1, x2, y2), w, h))

    return boxes

#### end

def fill_context_box(track, frame_idx):
    boxes = track["boxes"]
    if frame_idx in boxes:
        return boxes[frame_idx], 1
    prev_frames = [f for f in boxes if f < frame_idx]
    next_frames = [f for f in boxes if f > frame_idx]
    if prev_frames and next_frames:
        fp, fn = max(prev_frames), min(next_frames)
        alpha = float(frame_idx - fp) / float(fn - fp)
        bp = np.array(boxes[fp], dtype=np.float32)
        bn = np.array(boxes[fn], dtype=np.float32)
        interp = tuple(np.round((1 - alpha) * bp + alpha * bn).astype(np.int32).tolist())
        return interp, 0
    if prev_frames:
        return boxes[max(prev_frames)], 0
    if next_frames:
        return boxes[min(next_frames)], 0
    return None, 0


def crop_gray(gray_frame, box):
    x1, y1, x2, y2 = box
    crop = gray_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((112, 112), dtype=np.uint8)
    return cv2.resize(crop, (112, 112))


def overlap_count(a, b):
    return len(set(a["boxes"].keys()) & set(b["boxes"].keys()))


def build_inference_jobs(tracks, num_speakers):
    jobs = []
    for tid, target in tracks.items():
        frame_ids = sorted(target["boxes"].keys())
        if len(frame_ids) < 4:
            print ("len frame_ids less than 4")
            continue
        overlaps = sorted(
            [(overlap_count(target, other), oid)
             for oid, other in tracks.items()
             if oid != tid and overlap_count(target, other) > 0],
            reverse=True,
        )
        context_ids = [tid] + [oid for _, oid in overlaps[: max(0, num_speakers - 1)]]
        while len(context_ids) < num_speakers:
            context_ids.append(tid)
        jobs.append({"target_id": tid, "frame_ids": frame_ids, "context_ids": context_ids})
    return jobs


def list_audio_input_devices():
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                }
            )
    return devices


def list_pulse_input_sources():
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    devices = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            devices.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "details": "\t".join(parts[2:]),
                }
            )
    return devices


def list_webcams(max_index=8):
    devices = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ok, _ = cap.read()
        devices.append(
            {
                "index": idx,
                "width": width,
                "height": height,
                "fps": fps,
                "read_ok": bool(ok),
            }
        )
        cap.release()
    return devices


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def resolve_audio_input_config(device, target_sample_rate):
    info = sd.query_devices(device, "input")
    input_sr = int(round(info["default_samplerate"]))
    if input_sr <= 0:
        input_sr = int(target_sample_rate)
    input_channels = max(1, int(info["max_input_channels"]))
    return {
        "device": device,
        "name": info["name"],
        "input_sr": input_sr,
        "channels": min(2, input_channels),
    }


# ──────────────────────────── rolling audio buffer ───────────────────────────

class RollingAudioBuffer:
    """Thread-safe buffer that accumulates (timestamp, samples) audio chunks."""

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 30.0):
        self.sr = sample_rate
        self._max_seconds = max_seconds
        self._chunks: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def push(self, data: np.ndarray, wall_time: float):
        """Append a mono float32 chunk whose *start* time is wall_time."""
        chunk = data.flatten().astype(np.float32).copy()
        with self._lock:
            self._chunks.append((wall_time, chunk))
            cutoff = wall_time - self._max_seconds
            while self._chunks and self._chunks[0][0] < cutoff:
                self._chunks.popleft()

    def get_range(self, t_start: float, t_end: float) -> np.ndarray:
        """Return float32 mono samples covering [t_start, t_end] (wall clock)."""
        with self._lock:
            pieces = []
            for ts, chunk in self._chunks:
                chunk_end = ts + len(chunk) / self.sr
                if chunk_end <= t_start or ts >= t_end:
                    continue
                s0 = max(0, int((t_start - ts) * self.sr))
                s1 = min(len(chunk), int(np.ceil((t_end - ts) * self.sr)))
                if s1 > s0:
                    pieces.append(chunk[s0:s1])
        if pieces:
            return np.concatenate(pieces)
        return np.zeros(max(1, int((t_end - t_start) * self.sr)), dtype=np.float32)


class ParecAudioCapture:
    """Capture mono PCM audio from PulseAudio/PipeWire via parec."""

    def __init__(self, source, sample_rate, channels, frames_per_chunk, chunk_callback):
        self.source = source
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_chunk = int(frames_per_chunk)
        self.chunk_callback = chunk_callback
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        cmd = [
            "parec",
            "--raw",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
        ]
        if self.source:
            cmd.append(f"--device={self.source}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="talknce_parec",
        )
        self._thread.start()

    def _read_loop(self):
        bytes_per_frame = self.channels * 2
        chunk_bytes = self.frames_per_chunk * bytes_per_frame
        while not self._stop.is_set():
            if self._proc is None or self._proc.stdout is None:
                break
            raw = self._proc.stdout.read(chunk_bytes)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                continue
            if self.channels > 1:
                usable = (samples.size // self.channels) * self.channels
                if usable <= 0:
                    continue
                samples = samples[:usable].reshape(-1, self.channels).mean(axis=1)
            block_end = time.time()
            block_start = block_end - (len(samples) / max(1, self.sample_rate))
            self.chunk_callback(samples, self.sample_rate, block_start)

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def close(self):
        self.stop()


# ──────────────────────────── ROS 2 node ─────────────────────────────────────

class TalkNCELiveNode(Node):
    """
    Subscribes to a camera image topic, detects and tracks faces, runs
    TalkNCE active-speaker detection on a sliding buffer, and publishes
    annotated frames to an output image topic.

    Audio is captured from the system microphone and aligned with the video
    frames using monotonic wall-clock timestamps.
    """

    def __init__(self, args, cfg):
        super().__init__("talknce_live")

        # ── parameters ───────────────────────────────────────────────────────
        self.fps = float(args.fps)
        self.num_speakers = args.num_speakers
        self.score_thresh = args.score_thresh
        self.score_ema_alpha = DEFAULT_SCORE_EMA_ALPHA
        self.speak_off_thresh = max(0.0, self.score_thresh - DEFAULT_SPEAK_OFF_DELTA)
        self.buffer_seconds = float(args.buffer_seconds)
        self.audio_offset_s = float(args.audio_offset_ms) / 1000.0
        self.audio_offset_search_s = float(args.audio_offset_search_ms) / 1000.0
        self.audio_offset_step_s = max(0.01, float(args.audio_offset_step_ms) / 1000.0)
        self.frame_time_source = args.frame_time_source
        self.min_track_frames = args.min_track_frames
        self.max_track_miss = args.max_track_miss
        self.inference_stride = args.inference_stride
        self.iou_threshold = 0.25
        self.frame_headers = {}
        self.buffer_frames = int(self.buffer_seconds * self.fps)

        # Debugging / live-publish state.
        self.debug = args.debug
        self.debug_save_frames = args.debug_save_frames
        self.debug_dump_live_windows = args.debug_dump_live_windows
        self.debug_dump_live_limit = max(0, int(args.debug_dump_live_limit))
        self.debug_dump_count = 0
        self.debug_dump_live_dir = args.debug_dump_live_dir
        self.latest_frame = None
        self.latest_frame_idx = -1
        self.latest_header = None
        self.latest_scored_frame_idx = -1
        self.video_time_origin = None
        self.record_output = args.record_output
        self.record_writer = None
        self.record_fps = float(args.record_fps) if args.record_fps else 0.0
        self.record_path = os.path.abspath(args.record_output) if args.record_output else None
        self.debug_every_n = 60
        self.debug_frame_dir = "debug_live_frames"
        if self.debug_save_frames:
            os.makedirs(self.debug_frame_dir, exist_ok=True)
        if self.debug_dump_live_windows:
            os.makedirs(self.debug_dump_live_dir, exist_ok=True)
        self.latest_frame_lock = threading.Lock()

        # ── model (requires CUDA) ────────────────────────────────────────────
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TalkNCE inference requires CUDA. No GPU was found."
            )
        cfg.MODEL.NUM_SPEAKERS = self.num_speakers
        self.model_wrapper = loconet(cfg)
        self.model_wrapper.loadParameters(args.checkpoint)
        self.model_wrapper.eval()

        # ── face detector ────────────────────────────────────────────────────
        # cascade_path = os.path.join(
        #     cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        # )
        # self.detector = cv2.CascadeClassifier(cascade_path)
        # if self.detector.empty():
        #     raise RuntimeError("Failed to load Haar cascade face detector")

        # ── face detector (MediaPipe) ────────────────────────────────────────
        self.detect_every_n = max(1, int(args.detect_every_n))
        self.last_dets = []
        self.detector = create_mediapipe_face_detector(
            os.path.join(PROJECT_ROOT, "assets", "blaze_face_full_range_sparse.tflite")
        )

        # ── frame buffer  { global_frame_idx -> BGR np.ndarray } ─────────────
        self.frame_data: dict = {}
        self.frame_wall_times: dict = {}       # global_frame_idx -> monotonic
        self.frame_lock = threading.Lock()

        # ── tracking state (mutated only from the image callback thread) ──────
        # NOTE: rclpy.spin() uses a single thread by default, so no extra
        # lock is needed for the tracking state itself.  A snapshot is taken
        # under tracks_lock before the background inference thread reads it.
        self.active_tracks: dict = {}          # tid -> track_data
        self.next_track_id: int = 0
        self.global_frame_idx: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.tracks_lock = threading.Lock()    # guards active_tracks snapshot

        # ── score map  { (tid, global_frame_idx) -> probability } ────────────
        self.score_map: dict = {}
        self.latest_score_by_tid: dict = {}
        self.display_score_by_tid: dict = {}
        self.speaking_state_by_tid: dict = {}
        self.score_lock = threading.Lock()

        # ── audio buffer ─────────────────────────────────────────────────────
        self.audio_sr = 16000
        self.audio_backend = args.audio_backend
        self.audio_buf = RollingAudioBuffer(
            sample_rate=self.audio_sr, max_seconds=self.buffer_seconds + 5.0
        )
        self.audio_cfg = None
        self.audio_rms = 0.0
        self.audio_peak = 0.0
        self.audio_stats_lock = threading.Lock()
        self.audio_stream = None
        if self.audio_backend == "parec":
            input_sr = int(args.pulse_sample_rate)
            input_channels = max(1, int(args.pulse_channels))
            self.audio_cfg = {
                "device": args.audio_device,
                "name": args.audio_device or "default PulseAudio source",
                "input_sr": input_sr,
                "channels": input_channels,
            }
            self.audio_stream = ParecAudioCapture(
                source=args.audio_device,
                sample_rate=input_sr,
                channels=input_channels,
                frames_per_chunk=max(512, int(args.pulse_chunk_frames)),
                chunk_callback=self._audio_chunk_callback,
            )
            self.audio_stream.start()
            self.get_logger().info(
                "audio input backend=parec source={} stream_sr={} model_sr={}".format(
                    self.audio_cfg["name"],
                    self.audio_cfg["input_sr"],
                    self.audio_sr,
                )
            )
            self.get_logger().info(
                f"audio capture channels={self.audio_cfg['channels']}"
            )
        else:
            self.audio_cfg = resolve_audio_input_config(args.audio_device, self.audio_sr)
            self.audio_stream = sd.InputStream(
                device=self.audio_cfg["device"],
                samplerate=self.audio_cfg["input_sr"],
                channels=self.audio_cfg["channels"],
                dtype="float32",
                blocksize=512,
                callback=self._audio_callback,
            )
            self.audio_stream.start()
            self.get_logger().info(
                "audio input backend=sounddevice device={} ({}) stream_sr={} model_sr={}".format(
                    self.audio_stream.device,
                    self.audio_cfg["name"],
                    self.audio_cfg["input_sr"],
                    self.audio_sr,
                )
            )
            self.get_logger().info(
                f"audio capture channels={self.audio_cfg['channels']}"
            )

        # ── background inference thread ───────────────────────────────────────
        self._inference_pending = threading.Event()
        self._shutdown = threading.Event()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="talknce_infer"
        )
        self._inference_thread.start()

        # ── ROS 2 pub / sub ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.sub = None
        self.webcam_index = args.webcam_index
        self.webcam_cap = None
        self._camera_thread = None
        # Keep image callbacks serialized so MediaPipe video timestamps remain
        # strictly increasing, while allowing the publish timer to run in
        # parallel on a separate executor thread.
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.publish_callback_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(CompressedImage, args.output_topic, 10)
        self.speaker_bbox_pub = self.create_publisher(
            Detection2DArray, args.speaker_bbox_topic, 10
        )
        self.all_detections_pub = self.create_publisher(
            Detection2DArray, args.all_detections_topic, 10
        )
        
        if self.webcam_index is not None:
            self.webcam_cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_V4L2)
            if not self.webcam_cap.isOpened():
                self.webcam_cap.release()
                raise RuntimeError(f"Failed to open webcam index {self.webcam_index}")
            if args.webcam_width is not None:
                self.webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.webcam_width))
            if args.webcam_height is not None:
                self.webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.webcam_height))
            if args.webcam_fps is not None:
                self.webcam_cap.set(cv2.CAP_PROP_FPS, float(args.webcam_fps))
            self._camera_thread = threading.Thread(
                target=self._webcam_loop, daemon=True, name="talknce_webcam"
            )
            self._camera_thread.start()
            image_type = f"webcam[{self.webcam_index}]"
        else:
            self.image_is_compressed = args.image_topic.endswith("/compressed")
            if self.image_is_compressed:
                self.sub = self.create_subscription(
                    CompressedImage,
                    args.image_topic,
                    self._compressed_image_callback,
                    10,
                    callback_group=self.image_callback_group,
                )
                image_type = "sensor_msgs/CompressedImage"
            else:
                self.sub = self.create_subscription(
                    Image,
                    args.image_topic,
                    self._raw_image_callback,
                    10,
                    callback_group=self.image_callback_group,
                )
                image_type = "sensor_msgs/Image"
        self.publish_period = 1.0 / min(self.fps, 15.0)
        self.publish_timer = self.create_timer(
            self.publish_period,
            self._publish_latest_callback,
            callback_group=self.publish_callback_group,
        )

        self.get_logger().info(
            f"TalkNCE live node ready | "
            f"sub={args.image_topic if self.webcam_index is None else f'webcam[{self.webcam_index}]'} ({image_type}) pub={args.output_topic} | "
            f"fps={self.fps} buffer={self.buffer_seconds}s "
            f"stride={self.inference_stride}fr audio_offset_ms={args.audio_offset_ms} "
            f"frame_time_source={self.frame_time_source} "
            f"audio_offset_search_ms={args.audio_offset_search_ms}"
        )

    # ── audio capture ─────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        # sounddevice calls this from a C thread; keep it minimal.
        input_sr = self.audio_cfg["input_sr"]
        block_start = time.time() - frames / input_sr
        if indata.ndim == 2:
            chunk = np.mean(indata, axis=1)
        else:
            chunk = indata
        self._audio_chunk_callback(chunk, input_sr, block_start)

    def _audio_chunk_callback(self, chunk, input_sr, block_start):
        if input_sr != self.audio_sr:
            chunk = resampy.resample(chunk, input_sr, self.audio_sr)
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) > 0 else 0.0
        peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
        with self.audio_stats_lock:
            self.audio_rms = rms
            self.audio_peak = peak
        self.audio_buf.push(chunk, block_start)

    # ── image callback (ROS 2 spin thread) ───────────────────────────────────

    def _compressed_image_callback(self, msg: CompressedImage):
        # Decode compressed image (JPEG / PNG) directly with OpenCV.
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().error(
                f"cv2.imdecode failed (format='{msg.format}')"
            )
            return
        self._handle_frame(frame, msg.header)

    def _raw_image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge decode failed: {exc}")
            return
        self._handle_frame(frame, msg.header)

    def _webcam_loop(self):
        while not self._shutdown.is_set():
            ok, frame = self.webcam_cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = f"webcam_{self.webcam_index}"
            self._handle_frame(frame, header)

    def _handle_frame(self, frame: np.ndarray, header):
        arrival_t = time.time()
        header_t = None
        if header is not None:
            try:
                stamp = header.stamp
                stamp_secs = stamp_to_seconds(stamp)
                if stamp_secs > 0.0:
                    header_t = stamp_secs
            except Exception:
                header_t = None
        if self.frame_time_source == "header" and header_t is not None:
            wall_t = header_t
        else:
            wall_t = arrival_t
        if self.video_time_origin is None:
            self.video_time_origin = wall_t
        gidx = self.global_frame_idx
        h, w = frame.shape[:2]
        self.frame_width, self.frame_height = w, h

        if self.debug and gidx % self.debug_every_n == 0:
            self.get_logger().info(
                f"frame {gidx}: decoded image shape={w}x{h}"
            )

        t_decode = time.perf_counter()

        if self.debug_save_frames and gidx % 120 == 0:
            debug_path = os.path.join(self.debug_frame_dir, f"frame_{gidx:06d}.jpg")
            cv2.imwrite(debug_path, frame)
            self.get_logger().info(f"saved debug frame to {debug_path}")

        # ── face detection + tracking (single thread, no lock needed) ─────────
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # dets = detect_faces(gray, self.detector)
        # self._update_tracks(gidx, dets, w, h)

        # Mediapipe─ face detection + tracking (single thread, no lock needed) ─────────
        # timestamp_ms = int(wall_t * 1000)
        timestamp_ms = int(max(0.0, wall_t - self.video_time_origin) * 1000.0)

        if gidx % self.detect_every_n == 0:
            dets = detect_faces_mediapipe(frame, self.detector, timestamp_ms)
            self.last_dets = dets
        else:
            dets = self.last_dets

        if self.debug and gidx % self.debug_every_n == 0:
            with self.audio_stats_lock:
                audio_rms = self.audio_rms
                audio_peak = self.audio_peak
            self.get_logger().info(
                "frame {}: mediapipe detections={} audio_rms={:.4f} audio_peak={:.4f} "
                "frame_time={:.3f} arrival_time={:.3f} header_time={}".format(
                    gidx,
                    len(dets),
                    audio_rms,
                    audio_peak,
                    wall_t,
                    arrival_t,
                    "None" if header_t is None else f"{header_t:.3f}",
                )
            )

        with self.tracks_lock:
            self._update_tracks(gidx, dets, w, h)


        t_track = time.perf_counter()
        # ── store raw frame in sliding buffer ────────────────────────────────
        with self.frame_lock:
            self.frame_data[gidx] = frame.copy()
            self.frame_wall_times[gidx] = wall_t
            self.frame_headers[gidx] = header
            evict_before = gidx - self.buffer_frames
            for k in [k for k in list(self.frame_data) if k < evict_before]:
                del self.frame_data[k]
                self.frame_wall_times.pop(k, None)
                self.frame_headers.pop(k, None)

        self.global_frame_idx += 1
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()
            self.latest_frame_idx = gidx
            self.latest_header = header
        t_buf = time.perf_counter()

        # ── trigger inference every inference_stride frames ───────────────────
        if gidx % self.inference_stride == 0:
            self._inference_pending.set()
        t_pub = time.perf_counter()
    # ── face tracking (called from image callback, no extra lock) ────────────

    def _update_tracks(self, frame_idx: int, dets: list, w: int, h: int):
        """Greedy IoU matching then creation/eviction of face tracks."""
        assigned_tracks: set = set()
        assigned_dets: set = set()
        candidates = []
        for tid, tdata in self.active_tracks.items():
            lb = tdata["last_box"]
            for didx, dbox in enumerate(dets):
                s = iou_xyxy(lb, dbox)
                if s >= self.iou_threshold:
                    candidates.append((s, tid, didx))
        candidates.sort(reverse=True)

        for _, tid, didx in candidates:
            if tid in assigned_tracks or didx in assigned_dets:
                continue
            assigned_tracks.add(tid)
            assigned_dets.add(didx)
            box = clip_box(dets[didx], w, h)
            self.active_tracks[tid]["boxes"][frame_idx] = box
            self.active_tracks[tid]["last_box"] = box
            self.active_tracks[tid]["misses"] = 0

        for tid in list(self.active_tracks):
            if tid not in assigned_tracks:
                self.active_tracks[tid]["misses"] += 1

        # Remove stale tracks that have not been matched for too long.
        stale = [t for t, d in self.active_tracks.items()
                 if d["misses"] > self.max_track_miss]
        for tid in stale:
            del self.active_tracks[tid]

        # Start new tracks for unmatched detections.
        for didx, dbox in enumerate(dets):
            if didx in assigned_dets:
                continue
            box = clip_box(dbox, w, h)
            tid = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[tid] = {
                "id": tid,
                "boxes": {frame_idx: box},
                "last_box": box,
                "misses": 0,
            }

        # Evict old box entries that have scrolled out of the buffer window.
        evict_before = frame_idx - self.buffer_frames
        for tdata in self.active_tracks.values():
            for k in [k for k in list(tdata["boxes"]) if k < evict_before]:
                del tdata["boxes"][k]

    def _get_audio_clip_for_job(
        self,
        frame_ids_local,
        window_start,
        offset_s,
        effective_fps,
        anchor_fid,
        anchor_time,
    ):
        global_start_fid = window_start + frame_ids_local[0]
        global_end_fid = window_start + frame_ids_local[-1]
        fps = max(effective_fps, 1e-6)
        job_t0 = anchor_time + ((global_start_fid - anchor_fid) / fps) + offset_s
        job_t1 = anchor_time + ((global_end_fid + 1 - anchor_fid) / fps) + offset_s
        if job_t1 <= job_t0:
            return None
        audio_clip = self.audio_buf.get_range(job_t0, job_t1)
        if len(audio_clip) < int(0.2 * self.audio_sr):
            return None
        return audio_clip

    def _prepare_audio_for_vggish(self, audio_clip):
        # Match infer_raw_video.py, which feeds ffmpeg/wavfile int16-scale
        # samples into waveform_to_examples rather than normalized [-1, 1].
        scaled = np.asarray(audio_clip, dtype=np.float32) * 32768.0
        return np.clip(scaled, -32768.0, 32767.0)

    def _ensure_record_writer(self, frame):
        if self.record_path is None or self.record_writer is not None:
            return
        h, w = frame.shape[:2]
        fps = self.record_fps if self.record_fps > 0 else min(self.fps, 15.0)
        os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
        self.record_writer = cv2.VideoWriter(
            self.record_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not self.record_writer.isOpened():
            self.record_writer = None
            raise RuntimeError(f"Failed to open record output writer: {self.record_path}")
        self.get_logger().info(
            f"recording annotated live output to {self.record_path} @ {fps:.2f} fps"
        )

    def _record_annotated_frame(self, annotated):
        if self.record_path is None:
            return
        self._ensure_record_writer(annotated)
        if self.record_writer is not None:
            self.record_writer.write(annotated)

    def _infer_probs_for_job(self, audio_clip, visual_job, t, effective_fps):
        audio_clip = self._prepare_audio_for_vggish(audio_clip)
        audio_feat = vggish_input.waveform_to_examples(
            audio_clip, self.audio_sr, t, effective_fps, return_tensor=False
        )
        audio_feat = torch.FloatTensor(audio_feat)[None, None, :, :].cuda()
        visual_feat = torch.FloatTensor(visual_job)[None].cuda()

        with torch.no_grad(), torch.autocast('cuda'):
            b, s = 1, self.num_speakers
            visual_in = visual_feat.view(b * s, *visual_feat.shape[2:])
            audio_embed = self.model_wrapper.model.forward_audio_frontend(audio_feat)
            visual_embed = self.model_wrapper.model.forward_visual_frontend(visual_in)
            audio_embed = audio_embed.repeat(s, 1, 1)
            audio_embed, visual_embed = self.model_wrapper.model.forward_cross_attention(
                audio_embed, visual_embed
            )
            outs_av = self.model_wrapper.model.forward_audio_visual_backend(
                audio_embed, visual_embed, b, s
            )
            outs_av = outs_av.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
            logits = self.model_wrapper.lossAV.FC(outs_av)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        return probs

    def _estimate_effective_fps(self, frame_times):
        if len(frame_times) < 2:
            return self.fps
        diffs = np.diff(np.asarray(frame_times, dtype=np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-4)]
        if len(diffs) == 0:
            return self.fps
        median_dt = float(np.median(diffs))
        fps = 1.0 / median_dt
        return float(min(60.0, max(5.0, fps)))


    def _select_audio_offset(self, jobs, visuals, frame_times_snapshot, window_start):
        if self.audio_offset_search_s <= 0.0:
            return self.audio_offset_s

        candidates = []
        offset_s = self.audio_offset_s - self.audio_offset_search_s
        stop_s = self.audio_offset_s + self.audio_offset_search_s
        while offset_s <= stop_s + 1e-9:
            candidates.append(offset_s)
            offset_s += self.audio_offset_step_s

        for job_idx, job in enumerate(jobs):
            frame_ids_local = job["frame_ids"]
            t = len(frame_ids_local)
            if t < 1:
                continue
            global_frame_ids = [window_start + fid for fid in frame_ids_local]
            frame_times = [
                frame_times_snapshot[fid]
                for fid in global_frame_ids
                if fid in frame_times_snapshot
            ]
            if len(frame_times) < 2:
                continue
            effective_fps = self._estimate_effective_fps(frame_times)
            anchor_fid = global_frame_ids[0]
            anchor_time = frame_times[0]

            best_offset_s = self.audio_offset_s
            best_score = None
            for candidate_s in candidates:
                audio_clip = self._get_audio_clip_for_job(
                    frame_ids_local,
                    window_start,
                    candidate_s,
                    effective_fps,
                    anchor_fid,
                    anchor_time,
                )
                if audio_clip is None:
                    continue
                probs = self._infer_probs_for_job(audio_clip, visuals[job_idx], t, effective_fps)
                candidate_score = float(np.max(probs))
                if best_score is None or candidate_score > best_score:
                    best_score = candidate_score
                    best_offset_s = candidate_s

            if best_score is not None:
                if self.debug:
                    self.get_logger().info(
                        "inference: selected audio_offset_ms={:.1f} calib_score={:.3f}".format(
                            best_offset_s * 1000.0, best_score
                        )
                    )
                return best_offset_s

        return self.audio_offset_s

    def _dump_live_window(
        self,
        window_start,
        window_end,
        best_offset_s,
        window_audio,
        jobs,
        visuals,
        frame_times_snapshot,
        new_scores,
    ):
        if not self.debug_dump_live_windows:
            return
        if self.debug_dump_live_limit and self.debug_dump_count >= self.debug_dump_live_limit:
            return
        if not jobs:
            return

        job = jobs[0]
        target_id = job["target_id"]
        frame_ids_local = np.asarray(job["frame_ids"], dtype=np.int32)
        global_frame_ids = frame_ids_local + int(window_start)
        target_scores = np.asarray(
            [new_scores.get((target_id, int(fid)), np.nan) for fid in global_frame_ids],
            dtype=np.float32,
        )
        frame_times = np.asarray(
            [frame_times_snapshot.get(int(fid), np.nan) for fid in global_frame_ids],
            dtype=np.float64,
        )
        visual_job = np.asarray(visuals[0], dtype=np.uint8)

        dump_id = self.debug_dump_count
        stem = os.path.join(
            self.debug_dump_live_dir,
            f"live_window_{dump_id:03d}_g{window_start:06d}_{window_end:06d}",
        )

        np.savez_compressed(
            stem + ".npz",
            window_audio=np.asarray(window_audio, dtype=np.float32),
            audio_sr=np.int32(self.audio_sr),
            best_offset_s=np.float32(best_offset_s),
            frame_ids_local=frame_ids_local,
            global_frame_ids=global_frame_ids,
            frame_times=frame_times,
            target_scores=target_scores,
            visuals=visual_job,
            target_id=np.int32(target_id),
            window_start=np.int32(window_start),
            window_end=np.int32(window_end),
            fps=np.float32(self.fps),
        )

        contact_rows = []
        for speaker_idx in range(visual_job.shape[0]):
            row = np.concatenate([visual_job[speaker_idx, t] for t in range(visual_job.shape[1])], axis=1)
            contact_rows.append(row)
        if contact_rows:
            contact_sheet = np.concatenate(contact_rows, axis=0)
            cv2.imwrite(stem + "_crops.jpg", contact_sheet)

        meta = {
            "window_start": int(window_start),
            "window_end": int(window_end),
            "best_offset_ms": float(best_offset_s * 1000.0),
            "audio_sr": int(self.audio_sr),
            "target_id": int(target_id),
            "global_frame_ids": global_frame_ids.tolist(),
            "frame_times": frame_times.tolist(),
            "target_scores": target_scores.tolist(),
        }
        with open(stem + ".json", "w", encoding="ascii") as f:
            json.dump(meta, f, indent=2)

        self.debug_dump_count += 1
        self.get_logger().info(f"debug dump saved: {stem}.npz")

    # ── inference loop (background thread) ───────────────────────────────────

    def _inference_loop(self):
        while not self._shutdown.is_set():
            triggered = self._inference_pending.wait(timeout=1.0)
            if not triggered:
                continue
            self._inference_pending.clear()
            # try:
            self._run_inference()
            # except Exception as exc:
            #     self.get_logger().error(f"Inference error: {exc}")

    def _run_inference(self):
        t_inf0 = time.perf_counter() # timer
        # ── snapshot of global frame counter and track state ──────────────────
        gidx_now = self.global_frame_idx        # approximate; no lock needed for read
        window_end = gidx_now
        window_start = max(0, window_end - self.buffer_frames)

        # Build windowed track snapshot (local indices 0 … buffer_frames-1).
        with self.tracks_lock:
            tracks_snapshot = {}
            for tid, tdata in list(self.active_tracks.items()):
                local_boxes = {
                    fid - window_start: box
                    for fid, box in tdata["boxes"].items()
                    if window_start <= fid < window_end
                }
                if len(local_boxes) >= self.min_track_frames:
                    tracks_snapshot[tid] = {
                        "id": tid,
                        "boxes": local_boxes,
                        "last_box": tdata["last_box"],
                    }

        if not tracks_snapshot:
            if self.debug and window_end % 30 == 0:
                self.get_logger().info(
                    f"inference: no tracks_snapshot | window = [{window_start}, {window_end}]"
                )
            return
        # timer
        t_snap = time.perf_counter()
        if self.debug:
            self.get_logger().info(
                f"inference: snapshot prep {(t_snap - t_inf0)*1000:.1f} ms"
            )

        # ── snapshot of raw frames for this window ───────────────────────────
        # with self.frame_lock:
        #     frames_snapshot = {
        #         fid: self.frame_data[fid]
        #         for fid in range(window_start, window_end)
        #         if fid in self.frame_data
        #     }
        #     t_window_start = self.frame_wall_times.get(window_start)
        #     t_window_end = self.frame_wall_times.get(window_end - 1)
        with self.frame_lock:
            frames_snapshot = {
                fid: self.frame_data[fid]
                for fid in range(window_start, window_end)
                if fid in self.frame_data
            }
            frame_times_snapshot = {
                fid: self.frame_wall_times[fid]
                for fid in range(window_start, window_end)
                if fid in self.frame_wall_times
            }
            t_window_start = self.frame_wall_times.get(window_start)
            t_window_end = self.frame_wall_times.get(window_end - 1)

        if not frames_snapshot:
            return

        # Fall back to estimated times if the exact boundary frames are missing.
        now = time.monotonic()
        if t_window_end is None:
            t_window_end = now
        if t_window_start is None:
            t_window_start = t_window_end - self.buffer_frames / self.fps

        # ── fetch audio for the entire window as a float32 array in [-1, 1] ──
        # waveform_to_examples expects floats, not int16 (sounddevice already
        # delivers float32 in [-1, 1]).
        # ── build per-target inference jobs ──────────────────────────────────
        jobs = build_inference_jobs(tracks_snapshot, self.num_speakers)
        if not jobs:
            if self.debug and window_end % 30 == 0:
                self.get_logger().info(
                    f"inference: no jobs | tracks_snapshot={len(tracks_snapshot)}"
                )
            return

        fps = self.fps

        # ── pre-fill visual crops (one pass over the frame dictionary) ────────
        crop_requests: dict = defaultdict(list)
        for job_idx, job in enumerate(jobs):
            for s_idx, speaker_id in enumerate(job["context_ids"]):
                track = tracks_snapshot[speaker_id]
                for t_idx, local_fid in enumerate(job["frame_ids"]):
                    box, _ = fill_context_box(track, local_fid)
                    crop_requests[local_fid].append((job_idx, s_idx, t_idx, box))

        visuals = [
            np.zeros(
                (self.num_speakers, len(job["frame_ids"]), 112, 112),
                dtype=np.uint8,
            )
            for job in jobs
        ]
        for local_fid, requests in crop_requests.items():
            global_fid = local_fid + window_start
            frame = frames_snapshot.get(global_fid)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for job_idx, s_idx, t_idx, box in requests:
                crop = (
                    crop_gray(gray, box)
                    if box is not None
                    else np.zeros((112, 112), dtype=np.uint8)
                )
                visuals[job_idx][s_idx, t_idx] = crop
        
        # crop preparation
        t_crop = time.perf_counter()
        if self.debug:
            self.get_logger().info(
                f"inference: crop prep {(t_crop - t_snap)*1000:.1f} ms"
            )

        best_offset_s = self._select_audio_offset(
            jobs, visuals, frame_times_snapshot, window_start
        )
        audio_t_start = t_window_start + best_offset_s
        audio_t_end = t_window_end + best_offset_s
        window_audio = self.audio_buf.get_range(audio_t_start, audio_t_end)
        window_rms = float(np.sqrt(np.mean(np.square(window_audio)))) if len(window_audio) > 0 else 0.0
        window_peak = float(np.max(np.abs(window_audio))) if len(window_audio) > 0 else 0.0
        if self.debug and window_end % 30 == 0:
            self.get_logger().info(
                "inference: window_audio samples={} rms={:.4f} peak={:.4f}".format(
                    len(window_audio), window_rms, window_peak
                )
            )

        # ── run the model for each target track ───────────────────────────────
        new_scores: dict = {}
        for job_idx, job in enumerate(jobs):
            frame_ids_local = job["frame_ids"]
            t = len(frame_ids_local)
            if t < 1:
                continue
            global_frame_ids = [window_start + fid for fid in frame_ids_local]
            frame_times = [
                frame_times_snapshot[fid]
                for fid in global_frame_ids
                if fid in frame_times_snapshot
            ]
            if len(frame_times) < 2:
                continue
            effective_fps = self._estimate_effective_fps(frame_times)
            anchor_fid = global_frame_ids[0]
            anchor_time = frame_times[0]

            audio_clip = self._get_audio_clip_for_job(
                frame_ids_local,
                window_start,
                best_offset_s,
                effective_fps,
                anchor_fid,
                anchor_time,
            )
            if audio_clip is None:
                continue
            probs = self._infer_probs_for_job(audio_clip, visuals[job_idx], t, effective_fps)

            tid = job["target_id"]
            for idx, local_fid in enumerate(frame_ids_local):
                global_fid = local_fid + window_start
                new_scores[(tid, global_fid)] = float(probs[idx])

        # ── merge into shared score map, evict out-of-window entries ──────────
        # with self.score_lock:
        #     self.score_map.update(new_scores)
        #     for (tid, _), score in new_scores.items():
        #         self.latest_score_by_tid[tid] = score
        #     stale = [k for k in self.score_map if k[1] < window_start]
        #     for k in stale:
        #         del self.score_map[k]
        #     stale_tids = [
        #         tid for tid in list(self.latest_score_by_tid)
        #         if tid not in tracks_snapshot
        #     ]
        #     for tid in stale_tids:
        #         del self.latest_score_by_tid[tid]
        if self.debug and window_end % 30 == 0:
            self.get_logger().info(
                f"inference: produced {len(new_scores)} scores for window [{window_start}, {window_end}]"
            )
        if self.debug and new_scores:
            score_vals = np.array(list(new_scores.values()), dtype=np.float32)
            window_frame_times = [
                frame_times_snapshot[fid]
                for fid in sorted(frame_times_snapshot)
            ]
            effective_window_fps = self._estimate_effective_fps(window_frame_times)
            self.get_logger().info(
                "inference: score stats min={:.3f} mean={:.3f} max={:.3f} thresh={:.3f} | window_audio rms={:.4f} peak={:.4f} samples={} effective_fps={:.2f}".format(
                    float(score_vals.min()),
                    float(score_vals.mean()),
                    float(score_vals.max()),
                    self.score_thresh,
                    window_rms,
                    window_peak,
                    len(window_audio),
                    effective_window_fps,
                )
            )
        self._dump_live_window(
            window_start,
            window_end,
            best_offset_s,
            window_audio,
            jobs,
            visuals,
            frame_times_snapshot,
            new_scores,
        )
        with self.score_lock:
            self.score_map.update(new_scores)
            for (tid, fid), score in new_scores.items():
                prev = self.latest_score_by_tid.get(tid)
                if prev is None or fid >= prev[0]:
                    self.latest_score_by_tid[tid] = (fid, score)
                self._update_display_state(tid, fid, score)

            if new_scores:
                self.latest_scored_frame_idx = max(fid for (_, fid) in new_scores.keys())

            stale = [k for k in self.score_map if k[1] < window_start]
            for k in stale:
                del self.score_map[k]
            stale_tids = [tid for tid in list(self.latest_score_by_tid) if tid not in tracks_snapshot]
            for tid in stale_tids:
                del self.latest_score_by_tid[tid]
                self.display_score_by_tid.pop(tid, None)
                self.speaking_state_by_tid.pop(tid, None)

        # Timer
        t_inf1 = time.perf_counter()
        if self.debug:
            self.get_logger().info(
                f"inference: total update {(t_inf1 - t_inf0)*1000:.1f} ms"
            )

    def _update_display_state(self, tid, fid, raw_score):
        prev_display = self.display_score_by_tid.get(tid)
        if prev_display is None:
            smooth_score = float(raw_score)
        else:
            smooth_score = (
                self.score_ema_alpha * float(raw_score)
                + (1.0 - self.score_ema_alpha) * float(prev_display[1])
            )
        was_speaking = self.speaking_state_by_tid.get(tid, False)
        if was_speaking:
            speaking = smooth_score >= self.speak_off_thresh
        else:
            speaking = smooth_score >= self.score_thresh
        self.display_score_by_tid[tid] = (fid, smooth_score)
        self.speaking_state_by_tid[tid] = speaking
    def _publish_active_speaker_bboxes(self, gidx: int, header):
        """Publish Detection2DArray containing only currently-speaking tracks."""
        with self.score_lock:
            speaking_state = dict(self.speaking_state_by_tid)
        with self.tracks_lock:
            tracks_now = {
                tid: dict(tdata["boxes"])
                for tid, tdata in self.active_tracks.items()
            }

        array_msg = Detection2DArray()
        array_msg.header = header

        for tid, boxes in tracks_now.items():
            box = boxes.get(gidx)
            if box is None or not speaking_state.get(tid, False):
                continue
            x1, y1, x2, y2 = box
            det = Detection2D()
            det.header = header
            det.id = str(tid)
            # BoundingBox2D center is geometry_msgs/Pose2D on ROS2 Humble
            # (x, y directly); on Iron/Jazzy use det.bbox.center.position.x/y
            det.bbox.center.position.x = float(x1 + x2) / 2.0
            det.bbox.center.position.y = float(y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            array_msg.detections.append(det)

        self.speaker_bbox_pub.publish(array_msg)

    def _publish_all_detections(self, gidx: int, header):
        """Publish Detection2DArray of every tracked face visible at gidx."""
        with self.tracks_lock:
            tracks_now = {
                tid: dict(tdata["boxes"])
                for tid, tdata in self.active_tracks.items()
            }

        array_msg = Detection2DArray()
        array_msg.header = header

        for tid, boxes in tracks_now.items():
            box = boxes.get(gidx)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            det = Detection2D()
            det.header = header
            det.id = str(tid)
            det.bbox.center.position.x = float(x1 + x2) / 2.0
            det.bbox.center.position.y = float(y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            array_msg.detections.append(det)

        self.all_detections_pub.publish(array_msg)

    def _publish_latest_callback(self):
        with self.latest_frame_lock:
            if self.latest_frame is None or self.latest_header is None:
                if self.debug and self.global_frame_idx % self.debug_every_n == 0:
                    self.get_logger().info("publish timer: no latest frame/header available")
                return
            frame = self.latest_frame.copy()
            gidx = self.latest_frame_idx
            header = self.latest_header

        try:
            annotated = self._render_frame(frame, gidx)
            self._record_annotated_frame(annotated)
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out_msg = CompressedImage()
            out_msg.header = header
            out_msg.format = "jpeg"
            out_msg.data = buf.tobytes()
            self.pub.publish(out_msg)
            self._publish_active_speaker_bboxes(gidx, header)
            self._publish_all_detections(gidx, header)

            if self.debug and gidx % self.debug_every_n == 0:
                self.get_logger().info(
                    f"publish timer: published latest frame {gidx}"
                )
        except Exception as exc:
            self.get_logger().error(f"publish error: {exc}")
        ####
    # ── rendering (called from image callback, reads score_map atomically) ────

    # def _render_frame(self, frame: np.ndarray, gidx: int) -> np.ndarray:
    #     out = frame.copy()
    #     with self.score_lock:
    #         latest_scores = dict(self.latest_score_by_tid)
    #     with self.tracks_lock:
    #         tracks_now = {
    #             tid: {
    #                 "boxes": dict(tdata["boxes"]),
    #             }
    #             for tid, tdata in self.active_tracks.items()
    #         }

    #     for tid, tdata in tracks_now.items():
    #         box = tdata["boxes"].get(gidx)
    #         if box is None:
    #             continue

    #         x1, y1, x2, y2 = box
    #         score = latest_scores.get(tid)

    #         if score is None:
    #             # No inference result yet — draw a grey box.
    #             cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), 2)
    #             cv2.putText(
    #                 out, f"ID {tid}",
    #                 (x1, max(20, y1 - 8)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    #             )
    #         else:
    #             speaking = score >= self.score_thresh
    #             color = (0, 220, 0) if speaking else (0, 0, 220)
    #             cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    #             label = "ID {} | {:.2f} {}".format(
    #                 tid, score, "SPEAK" if speaking else "SILENT"
    #             )
    #             cv2.putText(
    #                 out, label,
    #                 (x1, max(20, y1 - 8)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    #             )
    #     return out
    def _render_frame(self, frame: np.ndarray, gidx: int) -> np.ndarray:
        out = frame.copy()
        with self.score_lock:
            score_map_now = dict(self.score_map)
            latest_score_by_tid = dict(self.latest_score_by_tid)
            display_score_by_tid = dict(self.display_score_by_tid)
            speaking_state_by_tid = dict(self.speaking_state_by_tid)
        with self.tracks_lock:
            tracks_now = {
                tid: {
                    "boxes": dict(tdata["boxes"]),
                }
                for tid, tdata in self.active_tracks.items()
            }

        for tid, tdata in tracks_now.items():
            box = tdata["boxes"].get(gidx)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            score = score_map_now.get((tid, gidx), None)
            if score is None:
                latest = display_score_by_tid.get(tid)
                if latest is not None:
                    score = latest[1]
                else:
                    latest_raw = latest_score_by_tid.get(tid)
                    if latest_raw is not None:
                        score = latest_raw[1]

            if score is None:
                cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), 2)
                cv2.putText(
                    out, f"ID {tid}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
                )
            else:
                speaking = speaking_state_by_tid.get(tid, score >= self.score_thresh)
                color = (0, 220, 0) if speaking else (0, 0, 220)
                label = "ID {} | {:.2f} {}".format(
                    tid, score, "SPEAK" if speaking else "SILENT"
                )
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out, label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
        return out

    # ── cleanup ───────────────────────────────────────────────────────────────

    # def destroy_node(self):
    #     self._shutdown.set()
    #     self._inference_pending.set()   # unblock the inference thread
    #     self._inference_thread.join(timeout=3.0)
    #     self.audio_stream.stop()
    #     self.audio_stream.close()
    #     super().destroy_node()
    #mediapipe
    def destroy_node(self):
        self._shutdown.set()
        self._inference_pending.set()   # unblock the inference thread
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
        self._inference_thread.join(timeout=3.0)
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
        if self.record_writer is not None:
            self.record_writer.release()
        if self.webcam_cap is not None:
            self.webcam_cap.release()
        self.detector.close()
        super().destroy_node()


# ──────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Live TalkNCE ASD via ROS 2 image topic + system microphone."
    )
    p.add_argument("--cfg", default=DEFAULT_CFG_PATH, help="Config yaml path.")
    p.add_argument(
        "--checkpoint",
        required=False,
        help="Pretrained model checkpoint. Required for live inference, not for --list-audio-devices.",
    )
    p.add_argument(
        "--image-topic", default=DEFAULT_IMAGE_TOPIC,
        help="Input ROS 2 compressed image topic (sensor_msgs/CompressedImage).",
    )
    p.add_argument(
        "--webcam-index", type=int, default=None,
        help="Read frames directly from a local webcam device index instead of a ROS image topic.",
    )
    p.add_argument(
        "--webcam-width", type=int, default=None,
        help="Optional webcam capture width.",
    )
    p.add_argument(
        "--webcam-height", type=int, default=None,
        help="Optional webcam capture height.",
    )
    p.add_argument(
        "--webcam-fps", type=float, default=None,
        help="Optional webcam capture FPS request.",
    )
    p.add_argument(
        "--output-topic", default=DEFAULT_OUTPUT_TOPIC,
        help="Output ROS 2 image topic with ASD bounding boxes.",
    )
    p.add_argument(
        "--speaker-bbox-topic", default=DEFAULT_SPEAKER_BBOX_TOPIC,
        help="Output ROS 2 topic (vision_msgs/Detection2DArray) for active-speaker bounding boxes only.",
    )
    p.add_argument(
        "--all-detections-topic", default=DEFAULT_ALL_DETECTIONS_TOPIC,
        help="Output ROS 2 topic (vision_msgs/Detection2DArray) for all tracked faces.",
    )
    p.add_argument(
        "--fps", type=float, default=DEFAULT_EXPECTED_FPS,
        help="Expected camera frame rate (used for audio alignment).",
    )
    p.add_argument("--num-speakers", type=int, default=3)
    p.add_argument(
        "--detect-every-n", type=int, default=DEFAULT_DETECT_EVERY_N,
        help="Run MediaPipe face detection every N frames. Use 1 for the most stable live tracking.",
    )
    p.add_argument(
        "--buffer-seconds", type=float, default=DEFAULT_BUFFER_SECONDS,
        help="Sliding window length in seconds fed to the model.",
    )
    p.add_argument(
        "--inference-stride", type=int, default=DEFAULT_INFERENCE_STRIDE,
        help="Run inference every N incoming frames (latency vs. CPU trade-off).",
    )
    p.add_argument("--min-track-frames", type=int, default=DEFAULT_MIN_TRACK_FRAMES,
                   help="Minimum track length to include in inference.")
    p.add_argument("--max-track-miss", type=int, default=DEFAULT_MAX_TRACK_MISS,
                   help="Max consecutive unmatched frames before a track is dropped.")
    p.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH,
                   help="Speaking probability threshold for the green/blue label.")
    p.add_argument(
        "--audio-offset-ms", type=float, default=DEFAULT_AUDIO_OFFSET_MS,
        help="Shift audio lookup relative to video timestamps. Positive values use later audio; negative values use earlier audio.",
    )
    p.add_argument(
        "--audio-offset-search-ms", type=float, default=DEFAULT_AUDIO_OFFSET_SEARCH_MS,
        help="If > 0, search +/- this range around --audio-offset-ms and pick the best offset online.",
    )
    p.add_argument(
        "--audio-offset-step-ms", type=float, default=DEFAULT_AUDIO_OFFSET_STEP_MS,
        help="Step size for online audio offset search.",
    )
    p.add_argument(
        "--frame-time-source", choices=["header", "arrival"], default=DEFAULT_FRAME_TIME_SOURCE,
        help="Use ROS header timestamps or local arrival time for video/audio alignment.",
    )
    p.add_argument("--debug", action = "store_true",
                   help="lightweight debugging")
    p.add_argument("--debug-save-frames", action="store_true",
                   help="Saving occasional debug frames")
    p.add_argument(
        "--debug-dump-live-windows", action="store_true",
        help="Save synchronized live inference windows for offline inspection.",
    )
    p.add_argument(
        "--debug-dump-live-dir", default="debug_live_windows",
        help="Directory for saved live inference window dumps.",
    )
    p.add_argument(
        "--debug-dump-live-limit", type=int, default=5,
        help="Maximum number of live inference windows to dump.",
    )
    p.add_argument(
        "--record-output",
        default="",
        help="Optional path to save the annotated live output as an mp4 while the node is running.",
    )
    p.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="Optional FPS for --record-output. Defaults to the live publish rate.",
    )
    p.add_argument(
        "--audio-device",
        default=DEFAULT_AUDIO_DEVICE,
        help="Audio input device. For sounddevice: PortAudio index or exact device name. For parec: PulseAudio source name.",
    )
    p.add_argument(
        "--audio-backend",
        choices=["sounddevice", "parec"],
        default=DEFAULT_AUDIO_BACKEND,
        help="Live audio capture backend. Use parec to match the working PulseAudio/PipeWire recording path.",
    )
    p.add_argument(
        "--pulse-sample-rate",
        type=int,
        default=DEFAULT_PULSE_SAMPLE_RATE,
        help="Input sample rate to request from parec.",
    )
    p.add_argument(
        "--pulse-channels",
        type=int,
        default=DEFAULT_PULSE_CHANNELS,
        help="Input channel count to request from parec.",
    )
    p.add_argument(
        "--pulse-chunk-frames",
        type=int,
        default=DEFAULT_PULSE_CHUNK_FRAMES,
        help="Frames per read from parec.",
    )
    p.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available input audio devices and exit.",
    )
    p.add_argument(
        "--list-pulse-sources",
        action="store_true",
        help="List available PulseAudio/PipeWire input sources and exit.",
    )
    p.add_argument(
        "--list-webcams",
        action="store_true",
        help="List local webcam indices that OpenCV can open and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_audio_devices:
        devices = list_audio_input_devices()
        if not devices:
            print("No input audio devices found.")
        else:
            print("Available input audio devices:")
            for dev in devices:
                print(
                    f"{dev['index']}: {dev['name']} | "
                    f"in={dev['max_input_channels']} | "
                    f"default_sr={dev['default_samplerate']}"
                )
        return

    if args.list_pulse_sources:
        devices = list_pulse_input_sources()
        if not devices:
            print("No PulseAudio/PipeWire input sources found.")
        else:
            print("Available PulseAudio/PipeWire input sources:")
            for dev in devices:
                print(f"{dev['index']}: {dev['name']} | {dev['details']}")
        return

    if args.list_webcams:
        devices = list_webcams()
        if not devices:
            print("No webcams found.")
        else:
            print("Available webcams:")
            for dev in devices:
                print(
                    f"{dev['index']}: read_ok={dev['read_ok']} | "
                    f"size={dev['width']}x{dev['height']} | fps={dev['fps']:.2f}"
                )
        return

    if not args.checkpoint:
        raise SystemExit(
            "error: --checkpoint is required unless --list-audio-devices, --list-pulse-sources, or --list-webcams is used"
        )

    # dlhammer.bootstrap re-parses sys.argv; shield it from this script's flags.
    orig_argv = sys.argv
    try:
        sys.argv = [orig_argv[0]]
        cfg = bootstrap(default_cfg={"cfg": args.cfg}, print_cfg=False)
    finally:
        sys.argv = orig_argv
    cfg.RESUME_PATH = args.checkpoint

    rclpy.init()
    node = TalkNCELiveNode(args, cfg)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
