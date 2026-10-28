from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

from jasna.main import build_parser
from jasna.mosaic.detections import Detections
from jasna.mosaic.head_filter import (
    POSE_INPUT_SIZE,
    POSE_MODEL_FILENAME,
    HeadFilteredDetector,
    PoseHeadMosaicFilter,
    _extract_person_head_and_hip_boxes,
)
from jasna.multi_gpu import (
    _build_cli_args_for_span,
    _split_long_render_spans,
    discover_logical_gpus,
    gpu_ltx_fast,
    physical_cuda_visible_token,
)
from jasna.session_config import SessionConfig


@dataclass(frozen=True)
class _KeyframeIndexStub:
    pts: tuple[int, ...]
    time_base: Fraction
    start_pts: int
    end_pts: int

    def seconds_for_pts(self, pts: int) -> float:
        return float((pts - self.start_pts) * self.time_base)


def _make_batch_detections(
    batch_boxes: list[list[tuple[int, int, int, int]]],
    *,
    frame_h: int = 1080,
    frame_w: int = 1920,
) -> Detections:
    boxes_xyxy: list[np.ndarray] = []
    masks: list[torch.Tensor] = []
    for boxes_list in batch_boxes:
        if not boxes_list:
            boxes_xyxy.append(np.zeros((0, 4), dtype=np.float32))
            masks.append(torch.zeros((0, frame_h, frame_w), dtype=torch.bool))
            continue
        boxes_xyxy.append(np.asarray(boxes_list, dtype=np.float32))
        m = torch.zeros((len(boxes_list), frame_h, frame_w), dtype=torch.bool)
        for idx, (x1, y1, x2, y2) in enumerate(boxes_list):
            m[idx, y1:y2, x1:x2] = True
        masks.append(m)
    return Detections(boxes_xyxy=boxes_xyxy, masks=masks)


def test_pose_head_filter_removes_head_mosaic_via_keypoints():
    filt = PoseHeadMosaicFilter(device="cpu")
    filt._load_attempted = True

    # Simulate COCO 17 keypoints for a standing male actor with face around (900, 170) and shoulders at y=280
    kpts = np.zeros((1, 17, 3), dtype=np.float32)
    kpts[0, 0] = [900.0, 170.0, 0.95]  # nose
    kpts[0, 1] = [880.0, 150.0, 0.95]  # left_eye
    kpts[0, 2] = [920.0, 150.0, 0.95]  # right_eye
    kpts[0, 5] = [810.0, 280.0, 0.95]  # left_shoulder
    kpts[0, 6] = [990.0, 280.0, 0.95]  # right_shoulder
    extracted = _extract_person_head_and_hip_boxes(kpts, 1080, 1920)
    filt._predict_keypoints = lambda _frames, _active, _hw: {0: extracted}

    # Frame has two mosaics:
    # 1) Male actor face mosaic at (830, 100, 970, 240)
    # 2) Core lower-body mosaic at (780, 620, 1120, 920)
    det = _make_batch_detections([[(830, 100, 970, 240), (780, 620, 1120, 920)]])
    frames = torch.zeros((1, 3, 1080, 1920), dtype=torch.uint8)

    filtered = filt.filter_detections(frames, det, target_hw=(1080, 1920))
    assert len(filtered.boxes_xyxy[0]) == 1
    assert tuple(int(v) for v in filtered.boxes_xyxy[0][0].tolist()) == (780, 620, 1120, 920)
    assert filtered.masks[0].shape[0] == 1


def test_pose_head_filter_never_removes_box_overlapping_hip_region():
    filt = PoseHeadMosaicFilter(device="cpu")
    filt._load_attempted = True

    # Simulate bent/close-interaction posture where head and hip are both near (820, 520)
    kpts = np.zeros((1, 17, 3), dtype=np.float32)
    kpts[0, 0] = [820.0, 480.0, 0.90]   # nose
    kpts[0, 11] = [780.0, 520.0, 0.95]  # left_hip
    kpts[0, 12] = [860.0, 520.0, 0.95]  # right_hip
    extracted = _extract_person_head_and_hip_boxes(kpts, 1080, 1920)
    filt._predict_keypoints = lambda _frames, _active, _hw: {0: extracted}

    det = _make_batch_detections([[(720, 420, 920, 620)]])
    frames = torch.zeros((1, 3, 1080, 1920), dtype=torch.uint8)

    filtered = filt.filter_detections(frames, det, target_hw=(1080, 1920))
    assert len(filtered.boxes_xyxy[0]) == 1
    assert tuple(int(v) for v in filtered.boxes_xyxy[0][0].tolist()) == (720, 420, 920, 620)


def test_geometric_dual_box_topology_fallback_removes_upper_head_mosaic():
    filt = PoseHeadMosaicFilter(device="cpu")
    filt._load_attempted = True
    # Simulate pose detector missing the occluded face
    filt._predict_keypoints = lambda _frames, _active, _hw: {0: ([], [])}

    # Upper small square face mosaic: cy = 180 (0.167 * H), w=140, h=140
    # Lower larger body mosaic: cy = 740 (0.685 * H), w=320, h=260
    det = _make_batch_detections([[(1200, 110, 1340, 250), (760, 610, 1080, 870)]])
    frames = torch.zeros((1, 3, 1080, 1920), dtype=torch.uint8)

    filtered = filt.filter_detections(frames, det, target_hw=(1080, 1920))
    assert len(filtered.boxes_xyxy[0]) == 1
    assert tuple(int(v) for v in filtered.boxes_xyxy[0][0].tolist()) == (760, 610, 1080, 870)


def test_head_filtered_detector_scan_scores_masks_zeros_head_only_frame():
    class DummyDetector:
        score_threshold = 0.25
        device = torch.device("cpu")

        def __call__(self, frames: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
            # When called on hit frames:
            # Frame 0: only a face mosaic -> should become empty after filter
            # Frame 1: face mosaic + body mosaic -> should retain body mosaic
            return _make_batch_detections(
                [
                    [(830, 100, 970, 240)],
                    [(830, 100, 970, 240), (760, 610, 1080, 870)],
                ],
                frame_h=target_hw[0],
                frame_w=target_hw[1],
            )

        def scan_scores_masks(self, frames: torch.Tensor, *, mask_hw: tuple[int, int]):
            b = frames.shape[0]
            scores = torch.full((b,), 0.92, dtype=torch.float32)
            masks = torch.ones((b, mask_hw[0], mask_hw[1]), dtype=torch.bool)
            return scores, masks

        def close(self) -> None:
            pass

    wrapper = HeadFilteredDetector(DummyDetector(), device="cpu")
    wrapper._filter._load_attempted = True
    kpts = np.zeros((1, 17, 3), dtype=np.float32)
    kpts[0, 0] = [900.0, 170.0, 0.95]
    kpts[0, 5] = [810.0, 280.0, 0.95]
    kpts[0, 6] = [990.0, 280.0, 0.95]
    extracted = _extract_person_head_and_hip_boxes(kpts, 1080, 1920)
    wrapper._filter._predict_keypoints = lambda _frames, active, _hw: {idx: extracted for idx in active}

    frames = torch.zeros((2, 3, 1080, 1920), dtype=torch.uint8)
    scores, masks = wrapper.scan_scores_masks(frames, mask_hw=(90, 160))

    assert scores.shape == (2,)
    assert masks.shape == (2, 90, 160)
    assert float(scores[0].item()) == 0.0
    assert int(masks[0].sum().item()) == 0
    assert float(scores[1].item()) > 0.9
    assert int(masks[1].sum().item()) > 0


def test_split_long_render_spans_balances_across_gpus():
    # 300 seconds video (keyframes every 2 seconds = 60 pts at time_base 1/30)
    keyframes = tuple(range(0, 9001, 60))
    index = _KeyframeIndexStub(keyframes, Fraction(1, 30), 0, 9000)
    raw_spans = [(0, 9000)]
    sample_times = tuple(float(t) for t in range(300))
    # Valley around 150s
    sample_scores = tuple(0.1 if 146 <= t <= 154 else 0.9 for t in range(300))

    split = _split_long_render_spans(
        raw_spans,
        index,
        num_gpus=2,
        sample_times=sample_times,
        sample_scores=sample_scores,
        threshold=0.25,
    )
    assert len(split) >= 2
    assert split[0][0] == 0
    assert split[-1][1] == 9000
    for a, b in zip(split, split[1:]):
        assert a[1] == b[0]
        assert a[1] in keyframes


def test_cli_multi_gpu_and_head_filter_flags():
    parser = build_parser()
    default_args = parser.parse_args(["--input", "in.mp4", "--output", "out.mp4"])
    assert default_args.multi_gpu is True
    assert default_args.ignore_head_mosaic is True
    assert default_args.gpus == ""

    custom_args = parser.parse_args(
        ["--input", "in.mp4", "--output", "out.mp4", "--no-multi-gpu", "--no-ignore-head-mosaic", "--gpus", "0,1"]
    )
    assert custom_args.multi_gpu is False
    assert custom_args.ignore_head_mosaic is False
    assert custom_args.gpus == "0,1"


def test_physical_cuda_visible_token_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert physical_cuda_visible_token(0) == "2"
    assert physical_cuda_visible_token(1) == "5"
    assert discover_logical_gpus("0,1") == [0] or discover_logical_gpus("0,1") == [0, 1]


def test_t6_pose_model_and_gpu_precision_selection(monkeypatch, tmp_path):
    from types import SimpleNamespace

    assert POSE_MODEL_FILENAME == "yolo26s-pose.pt"
    assert POSE_INPUT_SIZE == 640

    cfg = SimpleNamespace(
        ltx_fast=True,
        encoder_settings={"rc": "vbr", "maxrate": "5000k"},
        ltx_model="distilled2",
        restoration_model_name="ltx",
        batch_size=4,
        detection_model_name="rfdetr-v6-large",
        detection_model_path="model_weights/rfdetr-v6-large.onnx",
        detection_score_threshold=0.25,
        restoration_model_path="model_weights",
        codec="hevc",
        ltx_seed=42,
        fp16=True,
        ltx_large_canvas=False,
        ltx_window_hop=80,
        ltx_trial=False,
        lut_path=None,
        sharpen_strength=0.0,
        disable_progress=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Simulate GPU 0 = Blackwell (5090, supports NVFP4), GPU 1 = Ada (4090, INT8 only)
    monkeypatch.setattr(
        "jasna.multi_gpu.supports_nvfp4",
        lambda dev: str(dev) == "cuda:0",
    )
    assert gpu_ltx_fast(cfg, 0) is True
    assert gpu_ltx_fast(cfg, 1) is False

    span_in = tmp_path / "in.mp4"
    span_out = tmp_path / "out.mp4"
    args_5090 = _build_cli_args_for_span(cfg, span_in, span_out, tmp_path, ltx_fast=True)
    args_4090 = _build_cli_args_for_span(cfg, span_in, span_out, tmp_path, ltx_fast=False)
    assert "--ltx-fast" in args_5090
    assert "--ltx-fast" not in args_4090




def _chunk_case(intervals_sec, quiet_sec, duration_sec, *, num_gpus, max_inner_gap_sec=180.0):
    index = _KeyframeIndexStub(tuple(range(0, duration_sec * 30 + 1, 60)), Fraction(1, 30), 0, duration_sec * 30)
    sample_times = tuple(float(t) for t in range(duration_sec))
    sample_scores = tuple(
        0.1 if any(a <= t <= b for a, b in quiet_sec) else 0.9 for t in range(duration_sec)
    )
    return _split_long_render_spans(
        [(a * 30, b * 30) for a, b in intervals_sec],
        index,
        num_gpus=num_gpus,
        sample_times=sample_times,
        sample_scores=sample_scores,
        threshold=0.25,
        max_inner_gap_sec=max_inner_gap_sec,
    )


def test_render_chunks_never_cut_inside_running_mosaic():
    assert _chunk_case([(0, 300)], [], 300, num_gpus=2) == [(0, 9000)]


def test_render_chunks_group_short_mosaics_into_a_few_per_gpu():
    intervals = [(i * 100, i * 100 + 60) for i in range(36)]
    gaps = [(i * 100 + 61, i * 100 + 99) for i in range(36)]
    chunks = _chunk_case(intervals, gaps, 3600, num_gpus=4)
    # 2160 s of mosaic over 4 GPUs x 3 chunks -> 180 s per chunk, far fewer than 36 worker loads.
    assert 4 <= len(chunks) <= 13
    work = [sum(min(b * 30, end) - max(a * 30, start) for a, b in intervals if a * 30 < end and b * 30 > start)
            for start, end in chunks]
    assert max(work) <= 1.25 * sum(work) / 12
    starts = {a * 30 for a, _ in intervals}
    ends = {b * 30 for _, b in intervals}
    # Cuts only fall in mosaic-free gaps, so every chunk starts and ends on a mosaic boundary.
    assert all(start in starts and end in ends for start, end in chunks)


def test_render_chunks_split_on_long_gaps_only():
    intervals, gap = [(0, 100), (500, 600)], [(101, 499)]
    assert len(_chunk_case(intervals, gap, 700, num_gpus=1)) == 2
    assert _chunk_case(intervals, gap, 700, num_gpus=1, max_inner_gap_sec=600.0) == [(0, 18000)]
