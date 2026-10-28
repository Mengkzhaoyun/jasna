"""Filter out male-actor and passerby head/face mosaics using human pose keypoints.

Single-class mosaic detectors (RF-DETR / YOLO) detect any mosaic pixel grid on screen,
including censored male faces or bystanders. Restoring face mosaics with LTX wastes up
to 50% of 22B DiT compute on an extra track and produces uncanny facial artifacts.

``PoseHeadMosaicFilter`` runs a lightweight NMS-free YOLO26 pose model (``yolo26s-pose.pt``,
COCO-pose mAP50-95 63.0; falls back to a cached ``yolov8n-pose.pt``) only on frames where
mosaics were already detected:
1. Builds a ``HeadBox`` per person from head keypoints (0..4: nose, eyes, ears) and
   shoulder keypoints (5..6: left/right shoulder).
2. Builds a ``HipBox`` per person from hip keypoints (11..12: left/right hip).
3. Drops any candidate mosaic box that is covered by a ``HeadBox`` (>= 45%), is smaller
   than 35% of frame height, and is not covered by any ``HipBox`` (>= 12%, protecting
   close-up intimate interactions where a head is near a pelvis).
4. Without a matching ``HeadBox``, drops a small square box in the upper 36% of the frame
   when a clearly larger box sits below it (dual-mosaic topology fallback).
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jasna.engine_paths import model_weights_dir
from jasna.mosaic.detections import Detections

logger = logging.getLogger(__name__)

POSE_MODEL_FILENAME = "yolo26s-pose.pt"
# Older cached weights stay usable when the preferred model cannot be downloaded.
POSE_FALLBACK_FILENAMES = ("yolov8n-pose.pt",)
POSE_DOWNLOAD_URLS = (
    "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt",
)
POSE_INPUT_SIZE = 640

HEAD_MAX_SIDE_FRACTION = 0.35
HEAD_COVER_THRESHOLD = 0.45
HIP_PROTECT_COVER_THRESHOLD = 0.12
KP_CONF_THRESHOLD = 0.25


def is_head_filter_enabled() -> bool:
    env = os.environ.get("IGNORE_HEAD_MOSAIC", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _resolve_pose_weights_path() -> Path | None:
    custom = os.environ.get("POSE_MODEL_PATH", "").strip()
    if custom and Path(custom).is_file():
        return Path(custom)

    search_dirs = [
        Path("/data/jasna/model_weights"),
        Path("/app/sglang/model_weights"),
        model_weights_dir(),
        Path("model_weights"),
    ]
    for cand in (d / POSE_MODEL_FILENAME for d in search_dirs):
        if cand.is_file() and cand.stat().st_size > 1_000_000:
            return cand

    target_dir = Path("/data/jasna/model_weights") if Path("/data/jasna/model_weights").is_dir() else model_weights_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / POSE_MODEL_FILENAME
        partial = dest.with_suffix(".pt.part")
        for url in POSE_DOWNLOAD_URLS:
            try:
                logger.info("Downloading pose model for head-mosaic filtering from %s ...", url)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp, partial.open("wb") as out:
                    out.write(resp.read())
                if partial.stat().st_size > 1_000_000:
                    partial.replace(dest)
                    logger.info("Pose model saved to %s", dest)
                    return dest
            except Exception as exc:
                logger.debug("Pose model download attempt failed (%s): %s", url, exc)
            finally:
                partial.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not prepare pose model directory: %s", exc)
    for name in POSE_FALLBACK_FILENAMES:
        for cand in (d / name for d in search_dirs):
            if cand.is_file() and cand.stat().st_size > 1_000_000:
                logger.warning("Using fallback pose model %s; %s could not be downloaded", cand, POSE_MODEL_FILENAME)
                return cand
    return None


resolve_pose_model_path = _resolve_pose_weights_path


def _box_cover(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> float:
    """Fraction of ``inner`` area covered by ``outer``."""
    ix1 = max(outer[0], inner[0])
    iy1 = max(outer[1], inner[1])
    ix2 = min(outer[2], inner[2])
    iy2 = min(outer[3], inner[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if inner_area <= 1e-6:
        return 0.0
    return (iw * ih) / inner_area


def _extract_person_head_and_hip_boxes(
    kpts_xyc: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> tuple[list[tuple[tuple[float, float, float, float], float | None]], list[tuple[float, float, float, float]]]:
    """From COCO 17 keypoints ``(N, 17, 3)``, return ``(head_boxes_with_shoulder_y, hip_boxes)``."""
    head_boxes: list[tuple[tuple[float, float, float, float], float | None]] = []
    hip_boxes: list[tuple[float, float, float, float]] = []

    for person in kpts_xyc:
        if person.shape[0] < 13:
            continue
        head_kps = person[0:5]  # nose, l_eye, r_eye, l_ear, r_ear
        sh_kps = person[5:7]    # l_shoulder, r_shoulder
        hip_kps = person[11:13] # l_hip, r_hip

        valid_head = head_kps[head_kps[:, 2] >= KP_CONF_THRESHOLD]
        valid_sh = sh_kps[sh_kps[:, 2] >= KP_CONF_THRESHOLD]
        valid_hip = hip_kps[hip_kps[:, 2] >= KP_CONF_THRESHOLD]

        shoulder_max_y: float | None = None
        sh_span = 0.08 * frame_h
        if len(valid_sh) == 2:
            sh_span = max(float(abs(valid_sh[0, 0] - valid_sh[1, 0])), 0.07 * frame_h)
            shoulder_max_y = float(max(valid_sh[0, 1], valid_sh[1, 1]))
        elif len(valid_sh) == 1:
            shoulder_max_y = float(valid_sh[0, 1])

        if len(valid_head) >= 1:
            hx_min, hy_min = float(valid_head[:, 0].min()), float(valid_head[:, 1].min())
            hx_max, hy_max = float(valid_head[:, 0].max()), float(valid_head[:, 1].max())
            spread = max(hx_max - hx_min, hy_max - hy_min, 0.55 * sh_span, 0.06 * frame_h)
            pad_x = max(0.65 * spread, 0.35 * sh_span)
            pad_top = max(0.85 * spread, 0.45 * sh_span)
            pad_bot = max(0.55 * spread, 0.25 * sh_span)
            hx1 = max(0.0, hx_min - pad_x)
            hx2 = min(float(frame_w), hx_max + pad_x)
            hy1 = max(0.0, hy_min - pad_top)
            hy2 = min(float(frame_h), hy_max + pad_bot)
            if shoulder_max_y is not None:
                hy2 = min(hy2, shoulder_max_y + 0.06 * frame_h)
            if hx2 > hx1 and hy2 > hy1:
                head_boxes.append(((hx1, hy1, hx2, hy2), shoulder_max_y))
        elif len(valid_sh) == 2:
            # Face is heavily mosaicked so facial keypoints were suppressed, but both shoulders are clear
            sh_cx = float(0.5 * (valid_sh[0, 0] + valid_sh[1, 0]))
            sh_cy = float(0.5 * (valid_sh[0, 1] + valid_sh[1, 1]))
            head_r = max(0.55 * sh_span, 0.06 * frame_h)
            hx1 = max(0.0, sh_cx - head_r)
            hx2 = min(float(frame_w), sh_cx + head_r)
            hy1 = max(0.0, sh_cy - 1.55 * head_r)
            hy2 = min(float(frame_h), sh_cy + 0.15 * head_r)
            head_boxes.append(((hx1, hy1, hx2, hy2), shoulder_max_y))

        if len(valid_hip) >= 1:
            px_min, py_min = float(valid_hip[:, 0].min()), float(valid_hip[:, 1].min())
            px_max, py_max = float(valid_hip[:, 0].max()), float(valid_hip[:, 1].max())
            hip_r = max(float(px_max - px_min) * 0.85, 0.12 * frame_h)
            px1 = max(0.0, px_min - hip_r)
            px2 = min(float(frame_w), px_max + hip_r)
            py1 = max(0.0, py_min - 0.65 * hip_r)
            py2 = min(float(frame_h), py_max + 1.05 * hip_r)
            hip_boxes.append((px1, py1, px2, py2))

    return head_boxes, hip_boxes


def _is_head_mosaic_box(
    box: tuple[float, float, float, float],
    all_boxes: list[tuple[float, float, float, float]],
    head_boxes: list[tuple[tuple[float, float, float, float], float | None]],
    hip_boxes: list[tuple[float, float, float, float]],
    frame_h: int,
    frame_w: int,
) -> bool:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return False

    # 1. Size guard: never drop large foreground mosaics (>= 35% of frame height)
    if max(bw, bh) >= HEAD_MAX_SIDE_FRACTION * frame_h:
        return False

    # 2. Pelvic / close-interaction protection guard: keep if overlapping any hip region
    for hip_box in hip_boxes:
        if _box_cover(hip_box, box) >= HIP_PROTECT_COVER_THRESHOLD:
            return False

    cy = 0.5 * (y1 + y2)

    # 3. Pose HeadBox match
    for head_box, shoulder_max_y in head_boxes:
        if shoulder_max_y is not None and cy > shoulder_max_y + 0.05 * frame_h:
            continue
        if _box_cover(head_box, box) >= HEAD_COVER_THRESHOLD:
            return True

    # 4. Geometric topology fallback for dual-mosaic frames (upper small square head mosaic + lower main mosaic)
    aspect = bw / bh
    if cy < 0.36 * frame_h and 0.62 <= aspect <= 1.55 and max(bw, bh) < 0.20 * frame_h:
        box_area = bw * bh
        for other in all_boxes:
            if other is box:
                continue
            ox1, oy1, ox2, oy2 = other
            obw, obh = max(0.0, ox2 - ox1), max(0.0, oy2 - oy1)
            ocy = 0.5 * (oy1 + oy2)
            if ocy > 0.48 * frame_h and oy1 > y2 + 0.04 * frame_h and (obw * obh) >= 1.35 * box_area:
                return True

    return False


class PoseHeadMosaicFilter:
    """Lazy-loaded, thread-safe human pose filter that removes head mosaics from detections."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self._pose_model = None
        self._load_attempted = False
        self.filtered_boxes_count = 0
        self.kept_boxes_count = 0
        self.filtered_frames_count = 0

    def _ensure_pose_model(self):
        if self._load_attempted:
            return self._pose_model
        with self._lock:
            if self._load_attempted:
                return self._pose_model
            self._load_attempted = True
            weights = _resolve_pose_weights_path()
            if weights is None:
                logger.warning("Pose model weights unavailable; using geometric dual-box head-mosaic filter only")
                return None
            try:
                from ultralytics import YOLO

                self._pose_model = YOLO(str(weights))
                logger.info("Pose head-mosaic filter loaded: %s on %s", weights.name, self.device)
            except Exception as exc:
                logger.warning("Failed to load pose model (%s); falling back to geometric filter", exc)
                self._pose_model = None
            return self._pose_model

    @torch.inference_mode()
    def _predict_keypoints(
        self,
        frames_uint8_bchw: torch.Tensor,
        active_indices: list[int],
        target_hw: tuple[int, int],
    ) -> dict[int, tuple[list[tuple[tuple[float, float, float, float], float | None]], list[tuple[float, float, float, float]]]]:
        pose_model = self._ensure_pose_model()
        if pose_model is None or not active_indices:
            return {}

        th, tw = int(target_hw[0]), int(target_hw[1])
        fh, fw = int(frames_uint8_bchw.shape[-2]), int(frames_uint8_bchw.shape[-1])

        sub_batch = frames_uint8_bchw[active_indices]
        # Downscale on the GPU before the host copy: full-res 1080p/4K frames would otherwise
        # cross PCIe and be letterboxed on the CPU only to end up at POSE_INPUT_SIZE anyway.
        shrink = POSE_INPUT_SIZE / max(fh, fw)
        if shrink < 1.0:
            sh, sw = max(1, round(fh * shrink)), max(1, round(fw * shrink))
            sub_batch = F.interpolate(sub_batch.float(), size=(sh, sw), mode="area").round_().to(torch.uint8)
        else:
            sh, sw = fh, fw
        scale_x = tw / max(1.0, float(sw))
        scale_y = th / max(1.0, float(sh))

        # Ultralytics accepts BCHW float32 [0, 1] tensors or list of HWC numpy arrays
        imgs_np = [
            frame.permute(1, 2, 0).
            flip(-1).  # RGB -> BGR for Ultralytics
            contiguous().cpu().numpy()
            for frame in sub_batch
        ]
        out: dict[int, tuple[list[tuple[tuple[float, float, float, float], float | None]], list[tuple[float, float, float, float]]]] = {}
        try:
            with self._lock:
                results = pose_model.predict(
                    imgs_np,
                    device=str(self.device),
                    verbose=False,
                    conf=0.25,
                    imgsz=POSE_INPUT_SIZE,
                )
            for idx, res in zip(active_indices, results):
                if res.keypoints is None or res.keypoints.data is None or res.keypoints.data.numel() == 0:
                    out[idx] = ([], [])
                    continue
                kpts = res.keypoints.data.detach().cpu().numpy().copy()
                kpts[:, :, 0] *= scale_x
                kpts[:, :, 1] *= scale_y
                out[idx] = _extract_person_head_and_hip_boxes(kpts, th, tw)
        except Exception as exc:
            logger.debug("Pose inference skipped on batch: %s", exc)
        return out

    def filter_detections(
        self,
        frames_uint8_bchw: torch.Tensor,
        detections: Detections,
        *,
        target_hw: tuple[int, int],
    ) -> Detections:
        active_indices = [i for i, boxes in enumerate(detections.boxes_xyxy) if len(boxes) > 0]
        if not active_indices:
            return detections

        th, tw = int(target_hw[0]), int(target_hw[1])
        pose_map = self._predict_keypoints(frames_uint8_bchw, active_indices, (th, tw))

        new_boxes_list: list[np.ndarray] = []
        new_masks_list: list[torch.Tensor] = []
        for i, (boxes, masks) in enumerate(zip(detections.boxes_xyxy, detections.masks)):
            if len(boxes) == 0:
                new_boxes_list.append(boxes)
                new_masks_list.append(masks)
                continue
            head_boxes, hip_boxes = pose_map.get(i, ([], []))
            tuples = [tuple(float(v) for v in b.tolist()) for b in boxes]
            keep_indices = [
                j
                for j, b in enumerate(tuples)
                if not _is_head_mosaic_box(b, tuples, head_boxes, hip_boxes, th, tw)
            ]
            dropped = len(boxes) - len(keep_indices)
            if dropped > 0:
                with self._lock:
                    self.filtered_boxes_count += dropped
                    self.kept_boxes_count += len(keep_indices)
                    self.filtered_frames_count += 1
                    if self.filtered_frames_count == 1 or self.filtered_frames_count % 25 == 0:
                        logger.info(
                            "[HeadFilter %s] Filtered %d head/face mosaic boxes across %d frames (kept %d core mosaic boxes)",
                            self.device,
                            self.filtered_boxes_count,
                            self.filtered_frames_count,
                            self.kept_boxes_count,
                        )
            else:
                with self._lock:
                    self.kept_boxes_count += len(keep_indices)
            if len(keep_indices) == len(boxes):
                new_boxes_list.append(boxes)
                new_masks_list.append(masks)
            elif keep_indices:
                keep_arr = np.asarray(keep_indices, dtype=np.int64)
                keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=masks.device)
                new_boxes_list.append(boxes[keep_arr])
                new_masks_list.append(masks.index_select(0, keep_tensor))
            else:
                new_boxes_list.append(np.zeros((0, 4), dtype=boxes.dtype))
                new_masks_list.append(masks[:0])

        return Detections(boxes_xyxy=new_boxes_list, masks=new_masks_list)


class HeadFilteredDetector:
    """Wraps a mosaic detection model so both ``__call__`` and ``scan_scores_masks``
    automatically drop male/passerby head mosaics."""

    def __init__(self, detector, *, device: torch.device | str | None = None) -> None:
        self._detector = detector
        resolved_device = device if device is not None else getattr(detector, "device", "cuda:0")
        self._filter = PoseHeadMosaicFilter(resolved_device)

    @property
    def score_threshold(self) -> float:
        return float(getattr(self._detector, "score_threshold", 0.25))

    @property
    def device(self):
        return getattr(self._detector, "device", self._filter.device)

    def close(self) -> None:
        if hasattr(self._detector, "close"):
            self._detector.close()

    def __getattr__(self, name: str):
        return getattr(self._detector, name)

    def __call__(self, frames_uint8_bchw: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        detections = self._detector(frames_uint8_bchw, target_hw=target_hw)
        if not is_head_filter_enabled():
            return detections
        return self._filter.filter_detections(frames_uint8_bchw, detections, target_hw=target_hw)

    @torch.inference_mode()
    def scan_scores_masks(
        self, frames_uint8_bchw: torch.Tensor, *, mask_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores, merged_masks = self._detector.scan_scores_masks(frames_uint8_bchw, mask_hw=mask_hw)
        if not is_head_filter_enabled():
            return scores, merged_masks

        threshold = self.score_threshold
        hit_mask = scores >= threshold
        if not bool(hit_mask.any()):
            return scores, merged_masks

        hit_indices = torch.nonzero(hit_mask, as_tuple=False).flatten().tolist()
        fh, fw = int(frames_uint8_bchw.shape[-2]), int(frames_uint8_bchw.shape[-1])
        sub_frames = frames_uint8_bchw[hit_indices]
        filtered = self.__call__(sub_frames, target_hw=(fh, fw))

        scores = scores.clone()
        merged_masks = merged_masks.clone()
        for local_idx, batch_idx in enumerate(hit_indices):
            kept_masks = filtered.masks[local_idx]
            if len(kept_masks) == 0:
                scores[batch_idx] = 0.0
                merged_masks[batch_idx].zero_()
            else:
                combined = kept_masks.any(dim=0, keepdim=True)[None].float()
                resized = F.interpolate(combined, size=mask_hw, mode="area")[0, 0] > 0.0
                merged_masks[batch_idx] = resized
        return scores, merged_masks
