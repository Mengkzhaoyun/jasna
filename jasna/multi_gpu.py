"""Single-container Multi-GPU smart video restoration pipeline.

Implements the three-stage architecture approved in ``docs/MultiGpuSmartPipeline.md``:
1. Stage 1 (Scan & Keyframe Split): 1-fps whole-video mosaic scan with Pose head-mosaic
   filtering (``HeadFilteredDetector``), aligned to safe random-access keyframes and
   load-balanced into ``SpliceSpan`` chunks across all visible GPUs.
2. Stage 2 (Multi-GPU Dual-Channel Execution):
   - Channel A (Unrestored Spans): Skips LTX completely; fast-encodes via NVENC to the
     target bitrate (e.g. 5M HEVC) or stream-copies when codec & low-bitrate rules allow.
   - Channel B (Mosaic Render Spans): Runs frame-accurate ``scan_span`` with Pose
     head-mosaic filtering, then executes LTX restoration across multiple GPUs using
     both L1 (Span-level) and L2 (Track-level) parallelism.
3. Stage 3 (FFmpeg Lossless Assembly): Normalizes all fragments with keyframe extra-data
   headers, concatenates video streams losslessly, and remuxes original audio, subtitles,
   chapters, and container metadata.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from jasna.accelerator import AcceleratorVendor, supports_nvfp4, vendor_for_device
from jasna.job_resume import atomic_write, commit_file, job_fingerprints, read_json, resume_directory, write_json
from jasna.ltx.plan import TrackPlan, plans_from_arrays, plans_to_arrays
from jasna.media.encoder_settings import require_encodable_size
from jasna.mosaic.head_filter import HeadFilteredDetector, is_head_filter_enabled
from jasna.mosaic_scan import scan_mosaic_sizes, segments_from_scores
from jasna.os_utils import subprocess_no_window_kwargs
from jasna.progressbar import ProgressCallback
from jasna.segments import SegmentRange, normalize_segments
from jasna.session_config import SessionConfig

if TYPE_CHECKING:
    from jasna.media.probe import VideoMetadata
    from jasna.media.splice import KeyframeIndex, SpliceSpan
    from jasna.session_factory import RestorationSession

log = logging.getLogger(__name__)

WHOLE_VIDEO_PTS = (-(1 << 63), (1 << 63) - 1)
WHOLE_SPAN_KEY = f"{WHOLE_VIDEO_PTS[0]}-{WHOLE_VIDEO_PTS[1]}"
MIN_SPLIT_SPAN_SECONDS = 15.0
# Mosaic-free gaps up to this long stay inside one LTX render chunk (MULTI_GPU_MAX_INNER_GAP).
# 35s bridges brief scene transitions without swallowing multi-minute non-mosaic scenes.
DEFAULT_MAX_INNER_GAP_SECONDS = 35.0
# A 22B worker load costs ~30 s, so a few chunks per GPU balance the GPUs better than one
# chunk each; chunks keep at least this much mosaic time so loads stay a small share.
RENDER_CHUNKS_PER_GPU = 3
MIN_CHUNK_WORK_SECONDS = 120.0


def is_multi_gpu_pipeline_enabled() -> bool:
    """Whether the smart scan + head-filter + multi-GPU pipeline is enabled (default: True)."""
    env = os.environ.get("JASNA_SMART_PIPELINE", os.environ.get("MULTI_GPU", "")).strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


is_smart_pipeline_enabled = is_multi_gpu_pipeline_enabled


def discover_logical_gpus(requested_gpus: str | None = None) -> list[int]:
    """Return logical CUDA device indices ``[0, 1, ...]`` available inside this container."""
    if not torch.cuda.is_available():
        return [0]
    count = torch.cuda.device_count()
    if count <= 0:
        return [0]
    spec = (requested_gpus or os.environ.get("JASNA_GPUS") or "").strip()
    if not spec or spec.lower() == "all":
        return list(range(count))
    selected: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < count and idx not in selected:
                selected.append(idx)
    return selected or [0]


def physical_cuda_visible_token(logical_index: int) -> str:
    """Map PyTorch logical ``cuda:<logical_index>`` back to the host/container ``CUDA_VISIBLE_DEVICES`` token."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw.lower() not in {"all", "none", "void"}:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if 0 <= logical_index < len(tokens):
            return tokens[logical_index]
    return str(logical_index)


def gpu_ltx_fast(config: SessionConfig, logical_gpu: int) -> bool:
    """NVFP4 only runs on Blackwell; mixed RTX 4090 / 5090 hosts fall back to INT8 per GPU."""
    if not config.ltx_fast:
        return False
    return torch.cuda.is_available() and supports_nvfp4(torch.device(f"cuda:{logical_gpu}"))


def resolve_release_binary() -> Path | None:
    """Locate the official frozen Jasna binary if present (used for encrypted ``.safetensors.enc`` LTX weights)."""
    env_bin = os.environ.get("JASNA_RELEASE_BIN", "").strip()
    candidates = [
        Path(env_bin) if env_bin else None,
        Path("/app/jasna-release/jasna"),
        Path("/data/jasna/.release-cache/v0.12.0/jasna-linux-nvidia-0.12.0/jasna"),
        Path("/data/.release-cache/v0.12.0/jasna-linux-nvidia-0.12.0/jasna"),
        Path("/data/jasna/.release-cache/v0.11.3/jasna-linux-nvidia-0.11.3/jasna"),
        Path("/data/.release-cache/v0.11.3/jasna-linux-nvidia-0.11.3/jasna"),
    ]
    for cand in candidates:
        if cand is not None and cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def _extract_keyframe_span_nut(
    source: Path,
    span: SpliceSpan,
    index: KeyframeIndex,
    destination: Path,
    *,
    codec: str,
) -> None:
    """Fast keyframe-exact packet copy of ``[span.start_pts, span.end_pts)`` into a ``.nut`` file.

    Unlike ``create_copy_fragment`` in ``splice.py``, this stops demuxing as soon as 64
    packets past ``span.end_pts`` have been seen, avoiding scanning the remainder of a
    multi-hour video for every short span.
    """
    import av
    from jasna.media.splice import (
        _nal_framing,
        _next_access_unit_offset,
        _repacked,
        _run_ffmpeg,
        canonical_codec,
    )

    if destination.exists():
        destination.unlink()
    codec = canonical_codec(codec)
    if codec == "av1":
        start = index.seconds_for_pts(span.start_pts)
        duration = float((span.end_pts - span.start_pts) * index.time_base)
        args: list[str] = []
        if start > 0:
            args += ["-ss", f"{max(0.0, start):.9f}"]
        args += [
            "-i", str(source),
            "-t", f"{max(0.0, duration):.9f}",
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-avoid_negative_ts", "make_zero",
            "-f", "nut",
            str(destination),
        ]
        _run_ffmpeg(args, purpose=f"extract AV1 span at {start:.3f}s")
        return

    with (
        av.open(str(source), metadata_errors="replace") as src,
        av.open(str(destination), "w", format="nut") as dst,
    ):
        in_stream = src.streams.video[0]
        out_stream = dst.add_stream_from_template(in_stream)
        length_size, length_prefixed = _nal_framing(in_stream, codec)
        src.seek(span.start_pts, stream=in_stream, backward=True)
        carried = b""
        past_end_packets = 0
        for packet in src.demux(in_stream):
            if not packet.size:
                continue
            data = bytes(packet)
            cut = _next_access_unit_offset(
                data, codec=codec, length_size=length_size, length_prefixed=length_prefixed
            )
            access_unit, carried = carried + data[:cut], data[cut:]
            if packet.pts is None:
                continue
            if packet.pts >= span.end_pts:
                past_end_packets += 1
                if past_end_packets >= 64:
                    break
                continue
            if packet.pts < span.start_pts:
                continue
            if access_unit != data:
                packet = _repacked(packet, access_unit)
            packet.pts -= span.start_pts
            if packet.dts is not None:
                packet.dts -= span.start_pts
            packet.stream = out_stream
            dst.mux(packet)


def _extract_span_mp4(
    source: Path,
    span: SpliceSpan,
    index: KeyframeIndex,
    destination_mp4: Path,
    *,
    source_codec: str,
) -> None:
    """Losslessly extract keyframe-aligned ``span`` into a standalone ``.mp4`` file."""
    from jasna.media.splice import _run_ffmpeg

    temp_nut = destination_mp4.with_suffix(".tmp.nut")
    try:
        _extract_keyframe_span_nut(source, span, index, temp_nut, codec=source_codec)
        _run_ffmpeg(
            [
                "-i", str(temp_nut),
                "-map", "0:v:0",
                "-an",
                "-c:v", "copy",
                "-movflags", "+faststart",
                str(destination_mp4),
            ],
            purpose=f"remux extracted span {destination_mp4.name}",
        )
    finally:
        temp_nut.unlink(missing_ok=True)


def _max_inner_gap_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("MULTI_GPU_MAX_INNER_GAP", "")))
    except ValueError:
        return DEFAULT_MAX_INNER_GAP_SECONDS


def _quiet_keyframes(
    index: KeyframeIndex,
    sample_times: tuple[float, ...],
    sample_scores: tuple[float, ...],
    threshold: float,
) -> set[int]:
    """Keyframes whose neighbouring 1-fps samples are all below ``threshold`` (no mosaic on screen)."""
    quiet: set[int] = set()
    if not sample_times or not sample_scores:
        return quiet
    for kf_pts in index.pts:
        pos = bisect.bisect_left(sample_times, index.seconds_for_pts(kf_pts))
        nearby = sample_scores[max(0, pos - 1) : min(len(sample_scores), pos + 2)]
        if nearby and max(nearby) < threshold:
            quiet.add(kf_pts)
    return quiet


def _split_long_render_spans(
    render_intervals: list[tuple[int, int]],
    index: KeyframeIndex,
    *,
    num_gpus: int,
    sample_times: tuple[float, ...],
    sample_scores: tuple[float, ...],
    threshold: float,
    max_inner_gap_sec: float | None = None,
) -> list[tuple[int, int]]:
    """Group mosaic intervals into a few large render chunks, about ``RENDER_CHUNKS_PER_GPU`` per GPU.

    Each render chunk runs in one LTX worker that loads the model once for the whole chunk
    (scan first, then encode / denoise / compose every window, then release), so fewer,
    larger chunks mean fewer 22B model loads. Mosaic-free gaps up to ``max_inner_gap_sec``
    stay inside a chunk (the worker passes them through without LTX windows); longer gaps
    become NVENC fast-bypass spans. Chunks are only cut in mosaic-free gaps or at quiet
    keyframes, never in the middle of a running mosaic."""
    if not render_intervals:
        return []
    time_base = float(index.time_base)
    max_gap_pts = round((_max_inner_gap_seconds() if max_inner_gap_sec is None else max_inner_gap_sec) / time_base)
    margin_pts = round(MIN_SPLIT_SPAN_SECONDS * 0.6 / time_base)
    quiet = _quiet_keyframes(index, sample_times, sample_scores, threshold)

    def work(piece: list[tuple[int, int]]) -> int:
        return sum(e - s for s, e in piece)

    # 1. Absorb short mosaic-free gaps into the surrounding chunk.
    groups: list[list[tuple[int, int]]] = []
    for interval in sorted(render_intervals):
        if groups and interval[0] - groups[-1][-1][1] <= max_gap_pts:
            groups[-1].append(interval)
        else:
            groups.append([interval])

    # 2. Bisect heavy groups into chunks small enough for longest-first scheduling to even out.
    target = max(
        round(MIN_CHUNK_WORK_SECONDS / time_base),
        work([iv for g in groups for iv in g]) // max(1, num_gpus * RENDER_CHUNKS_PER_GPU),
    )

    def cuts(piece: list[tuple[int, int]]) -> list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]]:
        """Every allowed cut as ``(left_work, left, right)``."""
        found = []
        for i in range(1, len(piece)):
            found.append((work(piece[:i]), piece[:i], piece[i:]))
        done = 0
        for i, (s, e) in enumerate(piece):
            lo, hi = bisect.bisect_right(index.pts, s + margin_pts), bisect.bisect_left(index.pts, e - margin_pts)
            for kf in index.pts[lo:hi]:
                if kf in quiet:
                    found.append((done + kf - s, [*piece[:i], (s, kf)], [(kf, e), *piece[i + 1 :]]))
            done += e - s
        return found

    def split(piece: list[tuple[int, int]], parts: int) -> list[list[tuple[int, int]]]:
        if parts <= 1:
            return [piece]
        left_parts = parts // 2
        desired = work(piece) * left_parts / parts
        options = cuts(piece)
        if not options:
            return [piece]
        _, left, right = min(options, key=lambda c: abs(c[0] - desired))
        return split(left, left_parts) + split(right, parts - left_parts)

    chunks: list[tuple[int, int]] = []
    for group in groups:
        # A single GPU gains nothing from extra chunks, only extra model loads.
        parts = math.ceil(work(group) / target) if num_gpus > 1 and work(group) > target * 1.25 else 1
        for piece in split(group, parts):
            chunks.append((piece[0][0], piece[-1][1]))
    return chunks


def _parallel_scan_mosaic_sizes(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    metadata: VideoMetadata,
    index: KeyframeIndex,
    *,
    gpus: list[int],
    detector_for_gpu: Callable[[int], object],
):
    """Run Stage-1 1-fps mosaic scan across all visible GPUs in parallel by splitting keyframe intervals."""
    from jasna.media.video_decoder import VideoReader
    from jasna.mosaic_scan import SCAN_MASK_HW, MosaicSamples, mosaic_sizes, scan_sample_stride

    if len(gpus) <= 1 or len(index.pts) < len(gpus) * 2 or metadata.duration < 60.0:
        detector = session.detection_model_for(config)
        return scan_mosaic_sizes(
            detector,
            input_video,
            metadata,
            batch_size=config.batch_size,
            device=session.device,
            disable_progress=config.disable_progress,
            on_progress=lambda _frac: None,
        )

    frame_stride = scan_sample_stride(metadata.video_fps)
    time_base = float(metadata.time_base)
    kf_pts = index.pts
    num_workers = len(gpus)
    chunk_boundaries: list[tuple[int, int]] = []
    for i in range(num_workers):
        start_i = (len(kf_pts) * i) // num_workers
        end_i = (len(kf_pts) * (i + 1)) // num_workers
        c_start = index.start_pts if i == 0 else kf_pts[start_i]
        c_end = index.end_pts if i == num_workers - 1 else kf_pts[end_i]
        if c_end > c_start:
            chunk_boundaries.append((c_start, c_end))

    def _scan_chunk(gpu: int, start_pts: int, end_pts: int) -> tuple[list[float], list[float], list[float]]:
        dev = torch.device(f"cuda:{gpu}")
        if torch.cuda.is_available():
            torch.cuda.set_device(dev)
        det = detector_for_gpu(gpu)
        seek_ts = index.seconds_for_pts(start_pts) if start_pts > index.start_pts else None
        c_times: list[float] = []
        c_scores: list[float] = []
        c_sizes: list[float] = []
        batch_size = config.batch_size
        reader = VideoReader(str(input_video), batch_size, dev, metadata, frame_stride=frame_stride)
        with reader:
            for batch, pts_list in reader.frames(seek_ts=seek_ts):
                valid_pairs = [(idx, pts) for idx, pts in enumerate(pts_list) if start_pts <= pts < end_pts]
                if not valid_pairs:
                    if any(pts >= end_pts for pts in pts_list):
                        break
                    continue
                valid_indices = [idx for idx, _ in valid_pairs]
                valid_pts = [pts for _, pts in valid_pairs]
                sub_batch = batch[valid_indices] if len(valid_indices) < batch.shape[0] else batch
                count = len(valid_pts)
                if sub_batch.shape[0] < batch_size:
                    sub_batch = torch.cat(
                        (sub_batch, sub_batch[-1:].expand(batch_size - sub_batch.shape[0], -1, -1, -1))
                    )
                b_scores, b_masks = det.scan_scores_masks(sub_batch, mask_hw=SCAN_MASK_HW)
                c_times.extend(max(0.0, (pts - index.start_pts) * time_base) for pts in valid_pts)
                c_scores.extend(b_scores[:count].tolist())
                c_sizes.extend(
                    mosaic_sizes(
                        b_masks[:count].to(torch.uint8).cpu().numpy(),
                        frame_width=int(metadata.video_width),
                        frame_height=int(metadata.video_height),
                    )
                )
                if any(pts >= end_pts for pts in pts_list):
                    break
        return c_times, c_scores, c_sizes

    log.info(
        "[Stage 1] Parallel 1-fps scan across %d GPUs (%s) ...",
        len(chunk_boundaries),
        ", ".join(f"cuda:{g}" for g in gpus[: len(chunk_boundaries)]),
    )
    all_times: list[float] = []
    all_scores: list[float] = []
    all_sizes: list[float] = []
    with ThreadPoolExecutor(max_workers=len(chunk_boundaries), thread_name_prefix="Stage1Scan") as pool:
        futures = [
            pool.submit(_scan_chunk, gpu, c_start, c_end)
            for gpu, (c_start, c_end) in zip(gpus, chunk_boundaries)
        ]
        for fut in futures:
            ct, cs, cz = fut.result()
            all_times.extend(ct)
            all_scores.extend(cs)
            all_sizes.extend(cz)

    return MosaicSamples(
        tuple(all_times),
        tuple(all_scores),
        tuple(all_sizes),
        frame_stride / float(metadata.video_fps),
    )


def build_smart_multi_gpu_plan(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    metadata: VideoMetadata,
    index: KeyframeIndex,
    *,
    gpus: list[int],
    detector_for_gpu: Callable[[int], object],
    resume_root: Path,
) -> list[SpliceSpan]:
    """Stage 1: Run 1-fps mosaic scan (with Pose head filter) and build keyframe-aligned spans."""
    from jasna.media.splice import SpliceSpan, segment_render_span

    num_gpus = len(gpus)
    plan_file = resume_root / "multi_gpu_plan.json"
    stat = input_video.stat()
    base_key = {
        "input": str(input_video.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "detection_model": config.detection_model_name,
        "threshold": config.detection_score_threshold,
        "head_filter": is_head_filter_enabled(),
        "num_gpus": num_gpus,
    }
    plan_key = {
        **base_key,
        "layout": "task-chunks-v2",
        "max_inner_gap": _max_inner_gap_seconds(),
    }
    saved = read_json(plan_file)
    saved_key = saved.get("key") if isinstance(saved, dict) else None
    has_completed_segments = (resume_root / "segments").is_dir() and (
        any((resume_root / "segments").glob("*-raw.nut"))
        or any((resume_root / "segments").glob("0*.ts"))
        or any((resume_root / "segments").glob("0*.mkv"))
    )
    base_key_matches = isinstance(saved_key, dict) and all(
        saved_key.get(k) == v for k, v in base_key.items()
    )
    if saved is not None and (
        saved_key == plan_key or (base_key_matches and has_completed_segments)
    ):
        spans = [
            SpliceSpan(
                kind=item["kind"],
                start_pts=int(item["start_pts"]),
                end_pts=int(item["end_pts"]),
                effect_ranges=tuple(tuple(r) for r in item.get("effect_ranges", ())),
            )
            for item in saved["spans"]
        ]
        analysis_sec = float(saved.get("analysis_sec", 0.0))
        total_sec = float((index.end_pts - index.start_pts) * index.time_base)
        render_sec = sum(float((s.end_pts - s.start_pts) * index.time_base) for s in spans if s.is_render)
        skip_sec = max(0.0, total_sec - render_sec)
        log.info(
            "Reusing saved Stage-1 smart plan for %s (%d spans, %d render spans)",
            input_video.name,
            len(spans),
            sum(1 for s in spans if s.is_render),
        )
        from jasna.runtime_log import format_duration, get_active_task_log

        log.info(
            "[Stage 1 分析汇总] 分析时间=%s(复用缓存) 视频总长=%s 免处理时长=%s(%.1f分钟, 占比%.1f%%, 含无码与纯男优人头码剔除) 需LTX修复时长=%s(%.1f分钟, 占比%.1f%%)",
            format_duration(analysis_sec),
            format_duration(total_sec),
            format_duration(skip_sec),
            skip_sec / 60.0,
            (skip_sec / total_sec * 100.0) if total_sec > 0 else 0.0,
            format_duration(render_sec),
            render_sec / 60.0,
            (render_sec / total_sec * 100.0) if total_sec > 0 else 0.0,
        )
        active_task = get_active_task_log()
        if active_task is not None:
            active_task.stats["analysis_sec"] = analysis_sec
            active_task.stats["skip_sec"] = skip_sec
            active_task.stats["render_sec"] = render_sec
        return spans

    if saved is not None:
        # Fragments are named by span index, so ones from an older plan would land in the wrong place.
        log.info("Saved Stage-1 plan for %s is outdated; discarding its fragments", input_video.name)
        for stale in (resume_root / "segments", resume_root / "work"):
            shutil.rmtree(stale, ignore_errors=True)
        (resume_root / "segments").mkdir(parents=True, exist_ok=True)
        (resume_root / "work").mkdir(parents=True, exist_ok=True)

    log.info(
        "[Stage 1] Scanning %s (duration=%.1fs, fps=%.2f, head_filter=%s, gpus=%d) ...",
        input_video.name,
        metadata.duration,
        metadata.video_fps,
        "ON" if is_head_filter_enabled() else "OFF",
        num_gpus,
    )
    t0 = time.perf_counter()
    samples = _parallel_scan_mosaic_sizes(
        config,
        session,
        input_video,
        metadata,
        index,
        gpus=gpus,
        detector_for_gpu=detector_for_gpu,
    )
    analysis_sec = time.perf_counter() - t0
    log.info(
        "[Stage 1] 1-fps scan finished in %.2fs (%d samples, %d above threshold %.2f)",
        analysis_sec,
        len(samples.times),
        sum(1 for s in samples.scores if s >= config.detection_score_threshold),
        config.detection_score_threshold,
    )

    segments = segments_from_scores(
        samples.times,
        samples.scores,
        threshold=config.detection_score_threshold,
        stride=samples.stride,
        duration=metadata.duration,
    )

    if not segments:
        log.info(
            "[Stage 1] No target mosaics detected in %s after head-mosaic filtering; skipping LTX for entire video.",
            input_video.name,
        )
        spans = [SpliceSpan("reencode", index.start_pts, index.end_pts)]
    else:
        raw_render: list[tuple[int, int]] = []
        for seg in normalize_segments(segments, duration=metadata.duration):
            rspan = segment_render_span(seg, index)
            if raw_render and rspan.start_pts <= raw_render[-1][1]:
                raw_render[-1] = (raw_render[-1][0], max(raw_render[-1][1], rspan.end_pts))
            else:
                raw_render.append((rspan.start_pts, rspan.end_pts))

        balanced_render = _split_long_render_spans(
            raw_render,
            index,
            num_gpus=num_gpus,
            sample_times=samples.times,
            sample_scores=samples.scores,
            threshold=config.detection_score_threshold,
        )

        spans = []
        cursor = index.start_pts
        for r_start, r_end in balanced_render:
            if cursor < r_start:
                spans.append(SpliceSpan("reencode", cursor, r_start))
            spans.append(SpliceSpan("render", r_start, r_end, ((r_start, r_end),)))
            cursor = r_end
        if cursor < index.end_pts:
            spans.append(SpliceSpan("reencode", cursor, index.end_pts))

    total_sec = float((index.end_pts - index.start_pts) * index.time_base)
    render_sec = sum(float((s.end_pts - s.start_pts) * index.time_base) for s in spans if s.is_render)
    skip_sec = max(0.0, total_sec - render_sec)
    log.info(
        "[Stage 1] Plan ready: %d total spans (%d render spans = %.1fs [%.1f%%], %d fast-bypass spans = %.1fs [%.1f%%])",
        len(spans),
        sum(1 for s in spans if s.is_render),
        render_sec,
        (render_sec / total_sec * 100.0) if total_sec > 0 else 0.0,
        sum(1 for s in spans if not s.is_render),
        skip_sec,
        (skip_sec / total_sec * 100.0) if total_sec > 0 else 0.0,
    )
    from jasna.runtime_log import format_duration, get_active_task_log

    log.info(
        "[Stage 1 分析汇总] 分析时间=%s 视频总长=%s 免处理时长=%s(%.1f分钟, 占比%.1f%%, 含无码与纯男优人头码剔除) 需LTX修复时长=%s(%.1f分钟, 占比%.1f%%)",
        format_duration(analysis_sec),
        format_duration(total_sec),
        format_duration(skip_sec),
        skip_sec / 60.0,
        (skip_sec / total_sec * 100.0) if total_sec > 0 else 0.0,
        format_duration(render_sec),
        render_sec / 60.0,
        (render_sec / total_sec * 100.0) if total_sec > 0 else 0.0,
    )
    active_task = get_active_task_log()
    if active_task is not None:
        active_task.stats["analysis_sec"] = analysis_sec
        active_task.stats["skip_sec"] = skip_sec
        active_task.stats["render_sec"] = render_sec

    resume_root.mkdir(parents=True, exist_ok=True)
    write_json(
        plan_file,
        {
            "key": plan_key,
            "analysis_sec": round(analysis_sec, 3),
            "spans": [
                {
                    "kind": s.kind,
                    "start_pts": s.start_pts,
                    "end_pts": s.end_pts,
                    "effect_ranges": [list(r) for r in s.effect_ranges],
                }
                for s in spans
            ],
        },
    )
    return spans


def _encode_unrestored_span(
    config: SessionConfig,
    input_video: Path,
    metadata: VideoMetadata,
    index: KeyframeIndex,
    span: SpliceSpan,
    partial_nut: Path,
    final_nut: Path,
    *,
    device: torch.device,
    cancel_event: threading.Event,
) -> None:
    """Fast hardware NVENC re-encode of an unrestored span to match target codec & bitrate (e.g. HEVC 5M)."""
    from jasna.ltx.restore import video_frames
    from jasna.media.video_encoder import VideoEncoder

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
    partial_nut.unlink(missing_ok=True)
    encoder = VideoEncoder(
        str(partial_nut),
        device=device,
        metadata=metadata,
        codec=config.codec,
        encoder_settings=dict(config.encoder_settings),
        lut_path=config.lut_path,
        sharpen_strength=config.sharpen_strength,
        output_fps=metadata.video_fps_exact,
        pts_origin=span.start_pts,
        smart_fragment=True,
    )
    frames = video_frames(
        input_video,
        metadata,
        batch_size=config.batch_size,
        device=device,
        frame_stride=1,
        seek_ts=index.seconds_for_pts(span.start_pts),
        end_pts=span.end_pts,
    )
    entered = False
    try:
        for batch, pts in frames():
            if cancel_event.is_set():
                return
            if not entered:
                encoder.__enter__()
                entered = True
            for frame, frame_pts in zip(batch, pts):
                encoder.encode(frame, int(frame_pts))
    finally:
        if entered:
            encoder.__exit__(None, None, None)

    if not cancel_event.is_set() and partial_nut.is_file():
        with partial_nut.open("r+b") as stream:
            os.fsync(stream.fileno())
        commit_file(partial_nut, final_nut)


def _build_cli_args_for_span(
    config: SessionConfig,
    span_in: Path,
    span_out: Path,
    span_work_dir: Path,
    *,
    ltx_fast: bool,
) -> list[str]:
    """Build CLI arguments for ``$JASNA_RELEASE_BIN`` to restore one pre-scanned span."""
    encoder_str = ",".join(f"{k}={v}" for k, v in config.encoder_settings.items())
    model_flag = {
        "distilled2": "ltx",
        "distilled4": "ltx-4step",
        "distilled": "ltx-8step",
        "undistilled": "ltx-undistilled",
    }.get(config.ltx_model, "ltx") if config.restoration_model_name == "ltx" else "basicvsrpp"

    args = [
        "--input", str(span_in.resolve()),
        "--output", str(span_out.resolve()),
        "--working-directory", str(span_work_dir.resolve()),
        "--device", "cuda:0",
        "--batch-size", str(config.batch_size),
        "--detection-model", config.detection_model_name,
        "--detection-model-path", str(Path(config.detection_model_path).resolve()),
        "--detection-score-threshold", str(config.detection_score_threshold),
        "--restoration-model-name", model_flag,
        "--restoration-model-path", str(Path(config.restoration_model_path).resolve()),
        "--codec", config.codec,
        "--encoder-settings", encoder_str,
        "--ltx-seed", str(config.ltx_seed),
        "--log-level", "info",
    ]
    args.append("--fp16" if config.fp16 else "--no-fp16")
    if config.ltx_large_canvas:
        args.append("--ltx-large-canvas")
    if config.ltx_window_hop == 88:
        args.append("--no-experimental-ltx-less-overlap")
    if ltx_fast:
        args.append("--ltx-fast")
    if config.ltx_trial:
        args.append("--ltx-trial")
    if config.lut_path:
        args += ["--lut", str(config.lut_path)]
    if config.sharpen_strength > 0:
        args += ["--sharpen", str(config.sharpen_strength)]
    if config.disable_progress:
        args.append("--no-progress")
    return args


def _prescan_span_with_head_filter(
    config: SessionConfig,
    detector,
    span_in_mp4: Path,
    span_out_mp4: Path,
    span_work_dir: Path,
    *,
    device: torch.device,
    ltx_fast: bool,
    cancel_event: threading.Event,
) -> tuple[VideoMetadata, Path, list[list[TrackPlan]], int]:
    """Run Pass 1 ``scan_span`` on ``span_in_mp4`` using ``HeadFilteredDetector`` and save ``.plans.npz`` + ``job.json``."""
    from jasna.ltx.restore import LtxSegment, Progress as LtxProgress, large_canvas_fits, scan_span, video_frames
    from jasna.media.probe import get_video_meta_data

    span_meta = get_video_meta_data(str(span_in_mp4))
    worker_config = replace(
        config,
        device="cuda:0",
        working_dir=span_work_dir,
        ltx_auto_split_percent=None,
        ltx_fast=ltx_fast,
    )
    res_dir = resume_directory(worker_config, span_out_mp4) / "render"
    ltx_dir = res_dir / "ltx"
    ltx_dir.mkdir(parents=True, exist_ok=True)
    plan_npz = ltx_dir / f"{WHOLE_SPAN_KEY}.plans.npz"
    job_json = res_dir / "job.json"

    fingerprint, latents = job_fingerprints(worker_config, span_in_mp4, span_out_mp4, span_meta, ())
    if plan_npz.is_file() and job_json.is_file():
        saved_job = read_json(job_json)
        if saved_job is not None and saved_job.get("latents") == latents:
            with np.load(plan_npz, allow_pickle=False) as arrays:
                plans = plans_from_arrays(arrays)
                frames_count = int(arrays["frames"])
            return span_meta, res_dir, plans, frames_count

    free_bytes = torch.cuda.mem_get_info(device)[0]
    use_large_canvas = large_canvas_fits(config.ltx_large_canvas, free_bytes)
    frames_src = video_frames(
        span_in_mp4,
        span_meta,
        batch_size=config.batch_size,
        device=device,
        frame_stride=1,
        seek_ts=None,
        end_pts=None,
    )
    count = 0

    def counted_frames():
        nonlocal count
        with closing(frames_src()) as batches:
            for batch, pts in batches:
                count += len(pts)
                yield batch, pts

    progress = LtxProgress(max(1, span_meta.num_frames), disable=True, report=None)
    with progress.bar("scan", max(1, span_meta.num_frames)) as bar:
        span_plans = scan_span(
            counted_frames,
            (LtxSegment(*WHOLE_VIDEO_PTS, config.ltx_seed),),
            detector,
            batch_size=config.batch_size,
            frame_h=int(span_meta.video_height),
            frame_w=int(span_meta.video_width),
            large_canvas=use_large_canvas,
            window_hop=config.ltx_window_hop,
            bar=bar,
            cancel=cancel_event,
        )

    with atomic_write(plan_npz, "wb") as stream:
        np.savez(stream, frames=np.array(count), **plans_to_arrays(span_plans))
    write_json(
        job_json,
        {
            "fingerprint": fingerprint,
            "latents": latents,
            "input": str(span_in_mp4.resolve()),
            "output": str(span_out_mp4.resolve()),
        },
    )
    return span_meta, res_dir, span_plans, count


def _run_l2_collaborative_track_denoise(
    config: SessionConfig,
    worker_cmd_prefix: list[str],
    span_in_mp4: Path,
    span_meta: VideoMetadata,
    main_ltx_dir: Path,
    span_plans: list[list[TrackPlan]],
    frames_count: int,
    helper_gpus: list[int],
    *,
    cancel_event: threading.Event,
) -> None:
    """L2 Track-level Multi-GPU parallelism: split ``TrackPlan``s across multiple GPUs so each
    GPU encodes and denoises a disjoint subset of tracks into ``main_ltx_dir``."""
    tracks = list(span_plans[0]) if span_plans else []
    pending_tracks = [
        plan for plan in tracks
        if not (main_ltx_dir / f"{plan.track_id:06d}.{WHOLE_SPAN_KEY}-0.final.done").is_file()
    ]
    if len(pending_tracks) < 2 or len(helper_gpus) < 2:
        return

    num_workers = min(len(helper_gpus), len(pending_tracks))
    selected_gpus = helper_gpus[:num_workers]
    buckets: list[list[TrackPlan]] = [[] for _ in range(num_workers)]
    bucket_loads = [0] * num_workers
    # Greedy largest-first bin packing by window count
    for plan in sorted(pending_tracks, key=lambda p: len(p.windows), reverse=True):
        lightest = min(range(num_workers), key=lambda i: bucket_loads[i])
        buckets[lightest].append(plan)
        bucket_loads[lightest] += max(1, len(plan.windows))

    for b in buckets:
        b.sort(key=lambda p: p.track_id)

    log.info(
        "[Stage 2 L2] Distributing %d tracks (%d windows) across %d GPUs: %s",
        len(pending_tracks),
        sum(len(p.windows) for p in pending_tracks),
        num_workers,
        ", ".join(f"cuda:{g}({len(b)} tracks/{w} win)" for g, b, w in zip(selected_gpus, buckets, bucket_loads)),
    )

    def _run_helper(slot: int, logical_gpu: int, subset: list[TrackPlan]) -> None:
        if not subset or cancel_event.is_set():
            return
        helper_root = main_ltx_dir.parent.parent / f"l2_worker_{slot}_gpu{logical_gpu}"
        helper_out = helper_root / span_in_mp4.name
        helper_work = helper_root / "work"
        helper_work.mkdir(parents=True, exist_ok=True)
        helper_fast = gpu_ltx_fast(config, logical_gpu)
        worker_cfg = replace(
            config, device="cuda:0", working_dir=helper_work, ltx_auto_split_percent=None, ltx_fast=helper_fast
        )
        helper_res = resume_directory(worker_cfg, helper_out) / "render"
        helper_ltx = helper_res / "ltx"
        helper_ltx.mkdir(parents=True, exist_ok=True)

        fp, lat = job_fingerprints(worker_cfg, span_in_mp4, helper_out, span_meta, ())
        with atomic_write(helper_ltx / f"{WHOLE_SPAN_KEY}.plans.npz", "wb") as stream:
            np.savez(stream, frames=np.array(frames_count), **plans_to_arrays([subset]))
        write_json(
            helper_res / "job.json",
            {
                "fingerprint": fp,
                "latents": lat,
                "input": str(span_in_mp4.resolve()),
                "output": str(helper_out.resolve()),
            },
        )

        cmd = [
            *worker_cmd_prefix,
            *_build_cli_args_for_span(config, span_in_mp4, helper_out, helper_work, ltx_fast=helper_fast),
        ]
        env = os.environ.copy()
        env.pop("JASNA_MAIN_PID", None)
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("LC_ALL", "C.UTF-8")
        env["CUDA_VISIBLE_DEVICES"] = physical_cuda_visible_token(logical_gpu)
        env["JASNA_SMART_PIPELINE"] = "0"
        proc = subprocess.Popen(cmd, env=env, **subprocess_no_window_kwargs())
        try:
            while proc.poll() is None:
                if cancel_event.is_set():
                    proc.terminate()
                    break
                if all(
                    (helper_ltx / f"{p.track_id:06d}.{WHOLE_SPAN_KEY}-0.final.done").is_file()
                    for p in subset
                ):
                    # All assigned tracks have finished Pass 2 (encode) and Pass 3 (denoise)!
                    # Stop helper before it spends time on Pass 4 (compose).
                    proc.terminate()
                    break
                time.sleep(0.2)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            # Move completed track .final.pt and .final.done files into main_ltx_dir
            for p in subset:
                done_file = helper_ltx / f"{p.track_id:06d}.{WHOLE_SPAN_KEY}-0.final.done"
                if done_file.is_file():
                    for w in p.windows:
                        pt_file = helper_ltx / f"{w.index:06d}.{WHOLE_SPAN_KEY}-0.final.pt"
                        if pt_file.is_file():
                            commit_file(pt_file, main_ltx_dir / pt_file.name)
                    commit_file(done_file, main_ltx_dir / done_file.name)
            shutil.rmtree(helper_root, ignore_errors=True)

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="LtxL2Track") as pool:
        futures = [
            pool.submit(_run_helper, slot, gpu, subset)
            for slot, (gpu, subset) in enumerate(zip(selected_gpus, buckets))
        ]
        for fut in futures:
            fut.result()


def _render_mosaic_span_on_gpu(
    span_idx: int,
    total_spans: int,
    span: SpliceSpan,
    config: SessionConfig,
    input_video: Path,
    metadata: VideoMetadata,
    index: KeyframeIndex,
    span_work_dir: Path,
    partial_nut: Path,
    final_nut: Path,
    *,
    logical_gpu: int,
    detector_for_gpu: Callable[[int], object],
    release_bin: Path | None,
    idle_gpus_fn: Callable[[], list[int]],
    cancel_event: threading.Event,
) -> None:
    """Process one mosaic ``render`` span on ``cuda:<logical_gpu>`` (with optional L2 helper GPUs)."""
    device = torch.device(f"cuda:{logical_gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
    detector = detector_for_gpu(logical_gpu)
    ltx_fast = gpu_ltx_fast(config, logical_gpu)
    span_dir = span_work_dir / f"span_{span_idx:04d}"
    span_dir.mkdir(parents=True, exist_ok=True)
    span_in_mp4 = span_dir / f"span_{span_idx:04d}_in.mp4"
    span_out_mp4 = span_dir / f"span_{span_idx:04d}_out.mp4"

    start_sec = index.seconds_for_pts(span.start_pts)
    end_sec = index.seconds_for_pts(span.end_pts)
    log.info(
        "[Stage 2][GPU cuda:%d] Span %d/%d (%.1fs - %.1fs, duration=%.1fs): extracting & scanning ...",
        logical_gpu,
        span_idx + 1,
        total_spans,
        start_sec,
        end_sec,
        end_sec - start_sec,
    )

    if not span_in_mp4.is_file():
        from jasna.media.splice import canonical_codec

        _extract_span_mp4(
            input_video,
            span,
            index,
            span_in_mp4,
            source_codec=canonical_codec(metadata.codec_name),
        )

    if config.restoration_model_name == "ltx":
        supports_v012_resume = release_bin is None or "0.11." not in str(release_bin)
        if supports_v012_resume:
            span_meta, res_dir, span_plans, frames_count = _prescan_span_with_head_filter(
                config,
                detector,
                span_in_mp4,
                span_out_mp4,
                span_dir,
                device=device,
                ltx_fast=ltx_fast,
                cancel_event=cancel_event,
            )
            tracks = span_plans[0] if span_plans else []
            windows_count = sum(len(p.windows) for p in tracks)
            if windows_count == 0:
                log.info(
                    "[Stage 2][GPU cuda:%d] Span %d/%d: 0 target mosaic tracks after head-mosaic filter; fast-encoding via NVENC.",
                    logical_gpu,
                    span_idx + 1,
                    total_spans,
                )
                tmp_partial_nut = span_dir / f"{span_idx:04d}-partial.nut"
                tmp_raw_nut = span_dir / f"{span_idx:04d}-raw.nut"
                if logical_gpu == 0 or not torch.cuda.is_available():
                    _encode_unrestored_span(
                        config,
                        input_video,
                        metadata,
                        index,
                        span,
                        tmp_partial_nut,
                        tmp_raw_nut,
                        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        cancel_event=cancel_event,
                    )
                else:
                    task_file = span_dir / f"unrestored_span_{span_idx:04d}.json"
                    write_json(
                        task_file,
                        {
                            "input_video": str(input_video.resolve()),
                            "codec": config.codec,
                            "encoder_settings": dict(config.encoder_settings),
                            "lut_path": str(config.lut_path) if config.lut_path else None,
                            "sharpen_strength": float(config.sharpen_strength),
                            "batch_size": int(config.batch_size),
                            "spans": [
                                {
                                    "idx": span_idx,
                                    "start_pts": int(span.start_pts),
                                    "end_pts": int(span.end_pts),
                                    "partial_nut": str(tmp_partial_nut.resolve()),
                                    "final_nut": str(tmp_raw_nut.resolve()),
                                }
                            ],
                        },
                    )
                    env = os.environ.copy()
                    env["JASNA_WORKER_SUBPROCESS"] = "1"
                    env["CUDA_VISIBLE_DEVICES"] = physical_cuda_visible_token(logical_gpu)
                    res = subprocess.run(
                        [sys.executable, "-m", "jasna.multi_gpu", "--unrestored-worker", str(task_file)],
                        env=env,
                        check=False,
                    )
                    if res.returncode != 0:
                        raise RuntimeError(f"Fast-encode fallback failed on span {span_idx} (GPU cuda:{logical_gpu})")
                if not cancel_event.is_set() and tmp_raw_nut.is_file():
                    from jasna.media.splice import normalize_fragment

                    partial_nut.unlink(missing_ok=True)
                    normalize_fragment(tmp_raw_nut, partial_nut, codec=config.codec)
                    with partial_nut.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    commit_file(partial_nut, final_nut)
                shutil.rmtree(span_dir, ignore_errors=True)
                return

            log.info(
                "[Stage 2][GPU cuda:%d] Span %d/%d: restoring %d core tracks (%d windows) with LTX ...",
                logical_gpu,
                span_idx + 1,
                total_spans,
                len(tracks),
                windows_count,
            )

            worker_cmd_prefix = (
                [str(release_bin)]
                if (release_bin is not None and not config.ltx_trial)
                else [sys.executable, "-m", "jasna", "--no-multi-gpu"]
            )
            # Check if other GPUs are idle and can collaborate via L2 Track-level parallelism
            # Only pair GPUs of the same precision so tracks in one span share the same look
            extra_gpus = [
                g for g in idle_gpus_fn() if g != logical_gpu and gpu_ltx_fast(config, g) == ltx_fast
            ]
            if extra_gpus and len(tracks) >= 2:
                _run_l2_collaborative_track_denoise(
                    config,
                    worker_cmd_prefix,
                    span_in_mp4,
                    span_meta,
                    res_dir / "ltx",
                    span_plans,
                    frames_count,
                    [logical_gpu, *extra_gpus],
                    cancel_event=cancel_event,
                )
        else:
            log.info(
                "[Stage 2][GPU cuda:%d] Span %d/%d (%.1fs - %.1fs): restoring on physical GPU %s ...",
                logical_gpu,
                span_idx + 1,
                total_spans,
                start_sec,
                end_sec,
                physical_cuda_visible_token(logical_gpu),
            )

        if release_bin is not None and not config.ltx_trial:
            cmd = [
                str(release_bin),
                *_build_cli_args_for_span(config, span_in_mp4, span_out_mp4, span_dir, ltx_fast=ltx_fast),
            ]
            env = os.environ.copy()
            env.pop("JASNA_MAIN_PID", None)
            env["PYTHONIOENCODING"] = "utf-8"
            env.setdefault("LC_ALL", "C.UTF-8")
            env["CUDA_VISIBLE_DEVICES"] = physical_cuda_visible_token(logical_gpu)
            env["JASNA_SMART_PIPELINE"] = "0"
            env["JASNA_WORKER_SUBPROCESS"] = "1"
            completed = subprocess.run(cmd, env=env, check=False, **subprocess_no_window_kwargs())
            if completed.returncode != 0:
                raise RuntimeError(
                    f"LTX worker failed on span {span_idx} (GPU cuda:{logical_gpu}, exit code {completed.returncode})"
                )
        else:
            from jasna.session_factory import build_pipeline, build_restoration_session

            span_cfg = replace(
                config, device=str(device), working_dir=span_dir, ltx_auto_split_percent=None, ltx_fast=ltx_fast
            )
            span_session = build_restoration_session(span_cfg, log_callback=None)
            try:
                pipe = build_pipeline(span_cfg, span_session, span_in_mp4, span_out_mp4, segments=None, splice_plan=None)
                pipe.run()
            finally:
                span_session.close()
    else:
        # BasicVSR++ mode on assigned GPU with HeadFilteredDetector
        from jasna.session_factory import build_pipeline, build_restoration_session

        span_cfg = replace(config, device=str(device), working_dir=span_dir, ltx_auto_split_percent=None)
        span_session = build_restoration_session(span_cfg, log_callback=None)
        try:
            pipe = build_pipeline(span_cfg, span_session, span_in_mp4, span_out_mp4, segments=None, splice_plan=None)
            pipe.run()
        finally:
            span_session.close()

    if not cancel_event.is_set():
        if not span_out_mp4.is_file():
            raise RuntimeError(
                f"Worker completed without producing output for span {span_idx} on GPU cuda:{logical_gpu}: {span_out_mp4}"
            )
        from jasna.media.splice import normalize_fragment

        partial_nut.unlink(missing_ok=True)
        normalize_fragment(span_out_mp4, partial_nut, codec=config.codec)
        with partial_nut.open("r+b") as stream:
            os.fsync(stream.fileno())
        commit_file(partial_nut, final_nut)
        shutil.rmtree(span_dir, ignore_errors=True)


def _gpu_device_label(logical_gpu: int) -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(logical_gpu).replace(" ", "_")
        except Exception:
            pass
    return "GPU"


def run_smart_multi_gpu_video(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    *,
    requested_gpus: str | None = None,
) -> bool:
    """Run the 3-stage Smart Scan + Head Filter + Multi-GPU Restoration + FFmpeg Concat pipeline."""
    from collections import deque
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import concatenate_fragments, mux_final_output, normalize_fragment, probe_keyframes
    from jasna.mosaic.detection_registry import build_detection_model
    from jasna.runtime_log import format_duration, get_active_task_log

    metadata = get_video_meta_data(str(input_video))
    require_encodable_size(
        vendor_for_device(session.device),
        config.codec,
        metadata.video_width,
        metadata.video_height,
    )
    index = probe_keyframes(input_video, metadata)
    gpus = discover_logical_gpus(requested_gpus)
    num_gpus = len(gpus)
    release_bin = resolve_release_binary() if config.restoration_model_name == "ltx" else None

    if output_video.suffix.lower() != ".mp4":
        output_video = output_video.with_suffix(".mp4")

    log.info(
        "=== Jasna Multi-GPU Smart Pipeline: %s -> %s (GPUs=%s, codec=%s, head_filter=%s) ===",
        input_video.name,
        output_video.name,
        ",".join(f"cuda:{g}(phys:{physical_cuda_visible_token(g)})" for g in gpus),
        config.codec,
        "ON" if is_head_filter_enabled() else "OFF",
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    resume_root = resume_directory(config, output_video)
    segments_dir = resume_root / "multi_gpu" / "segments"
    work_dir = resume_root / "multi_gpu" / "work"
    segments_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cancel_event = threading.Event()

    # Per-GPU detector cache for Stage 1 parallel scan and Stage 2 frame-level scan
    detectors_lock = threading.Lock()
    gpu_detectors: dict[int, object] = {}

    def detector_for_gpu(logical_gpu: int):
        if logical_gpu == 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            return session.detection_model_for(config)
        with detectors_lock:
            det = gpu_detectors.get(logical_gpu)
            if det is None:
                dev = torch.device(f"cuda:{logical_gpu}")
                if torch.cuda.is_available():
                    torch.cuda.set_device(dev)
                raw_det = build_detection_model(
                    config.detection_model_name,
                    config.detection_model_path,
                    batch_size=config.batch_size,
                    device=dev,
                    score_threshold=config.detection_score_threshold,
                    fp16=config.fp16,
                )
                det = HeadFilteredDetector(raw_det, device=dev) if is_head_filter_enabled() else raw_det
                gpu_detectors[logical_gpu] = det
            return det

    # Stage 1: Whole-video 1-fps scan + Pose head filter + keyframe splice plan
    spans = build_smart_multi_gpu_plan(
        config,
        session,
        input_video,
        metadata,
        index,
        gpus=gpus,
        detector_for_gpu=detector_for_gpu,
        resume_root=resume_root / "multi_gpu",
    )

    fragment_suffix = ".ts" if config.codec in {"h264", "hevc"} else ".mkv"

    def raw_path(span_idx: int) -> Path:
        return segments_dir / f"{span_idx:04d}-raw.nut"

    def partial_path(span_idx: int) -> Path:
        return segments_dir / f"{span_idx:04d}-partial.nut"

    def norm_path(span_idx: int) -> Path:
        return segments_dir / f"{span_idx:04d}{fragment_suffix}"

    def partial_norm_path(span_idx: int) -> Path:
        return segments_dir / f"{span_idx:04d}-partial{fragment_suffix}"

    def is_span_done(span_idx: int) -> bool:
        return norm_path(span_idx).is_file() or raw_path(span_idx).is_file()

    def span_duration(span: SpliceSpan) -> float:
        return float((span.end_pts - span.start_pts) * index.time_base)

    total_spans = len(spans)
    t_stage2_start = time.perf_counter()

    # Stage 2: Execute mosaic render spans (Channel B) and unrestored spans (Channel A) across all GPUs.
    # Unrestored spans are pre-partitioned per GPU and executed automatically as each GPU enters its
    # tail idle window after finishing its last LTX render chunk, hiding Channel A latency.
    pending_unrestored = [
        (idx, span) for idx, span in enumerate(spans)
        if not span.is_render and not is_span_done(idx)
    ]
    pending_render = [
        (idx, span) for idx, span in enumerate(spans)
        if span.is_render and not is_span_done(idx)
    ]
    all_reused = (len(pending_unrestored) == 0 and len(pending_render) == 0)
    # Sort render spans longest-first (LPT scheduling) to minimize tail latency across GPUs
    pending_render.sort(key=lambda item: item[1].end_pts - item[1].start_pts, reverse=True)

    unrestored_buckets: dict[int, list[tuple[int, SpliceSpan]]] = {g: [] for g in gpus}
    bucket_loads: dict[int, int] = {g: 0 for g in gpus}
    for idx, span in sorted(
        pending_unrestored,
        key=lambda item: item[1].end_pts - item[1].start_pts,
        reverse=True,
    ):
        target_gpu = min(gpus, key=lambda g: bucket_loads[g])
        unrestored_buckets[target_gpu].append((idx, span))
        bucket_loads[target_gpu] += span.end_pts - span.start_pts

    busy_gpus_lock = threading.Lock()
    busy_gpus: set[int] = set()
    render_queue_lock = threading.Lock()
    render_queue: deque[tuple[int, SpliceSpan]] = deque(pending_render)

    gpu_stats_lock = threading.Lock()
    gpu_render_stats: dict[int, dict[str, float | int]] = {
        g: {"spans": 0, "video_sec": 0.0, "wall_sec": 0.0} for g in gpus
    }
    ch_a_wall_times: list[float] = []

    def current_idle_gpus() -> list[int]:
        with busy_gpus_lock:
            return [g for g in gpus if g not in busy_gpus]

    def _run_unrestored_bucket_for_gpu(gpu: int) -> None:
        items = unrestored_buckets.get(gpu) or []
        if not items or cancel_event.is_set():
            return
        t_u0 = time.perf_counter()
        items.sort(key=lambda item: item[0])
        if num_gpus <= 1 or not torch.cuda.is_available():
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for idx, span in items:
                if cancel_event.is_set():
                    break
                _encode_unrestored_span(
                    config,
                    input_video,
                    metadata,
                    index,
                    span,
                    partial_path(idx),
                    raw_path(idx),
                    device=dev,
                    cancel_event=cancel_event,
                )
        else:
            task_file = work_dir / f"unrestored_gpu_{gpu}.json"
            write_json(
                task_file,
                {
                    "input_video": str(input_video.resolve()),
                    "codec": config.codec,
                    "encoder_settings": dict(config.encoder_settings),
                    "lut_path": str(config.lut_path) if config.lut_path else None,
                    "sharpen_strength": float(config.sharpen_strength),
                    "batch_size": int(config.batch_size),
                    "spans": [
                        {
                            "idx": idx,
                            "start_pts": int(s.start_pts),
                            "end_pts": int(s.end_pts),
                            "partial_nut": str(partial_path(idx).resolve()),
                            "final_nut": str(raw_path(idx).resolve()),
                        }
                        for idx, s in items
                    ],
                },
            )
            phys_token = physical_cuda_visible_token(gpu)
            env = os.environ.copy()
            env["JASNA_WORKER_SUBPROCESS"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = phys_token
            cmd = [sys.executable, "-m", "jasna.multi_gpu", "--unrestored-worker", str(task_file)]
            res = subprocess.run(cmd, env=env, check=False)
            task_file.unlink(missing_ok=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Unrestored fast-bypass worker on GPU cuda:{gpu} (phys:{phys_token}) exited with code {res.returncode}"
                )
        with gpu_stats_lock:
            ch_a_wall_times.append(time.perf_counter() - t_u0)

    def _gpu_worker_loop(gpu: int) -> None:
        while not cancel_event.is_set():
            with render_queue_lock:
                if not render_queue:
                    break
                idx, span = render_queue.popleft()
            with busy_gpus_lock:
                busy_gpus.add(gpu)
            t_r0 = time.perf_counter()
            try:
                _render_mosaic_span_on_gpu(
                    idx,
                    total_spans,
                    span,
                    config,
                    input_video,
                    metadata,
                    index,
                    work_dir,
                    partial_norm_path(idx),
                    norm_path(idx),
                    logical_gpu=gpu,
                    detector_for_gpu=detector_for_gpu,
                    release_bin=release_bin,
                    idle_gpus_fn=current_idle_gpus,
                    cancel_event=cancel_event,
                )
            finally:
                dt = time.perf_counter() - t_r0
                with busy_gpus_lock:
                    busy_gpus.discard(gpu)
            with gpu_stats_lock:
                gpu_render_stats[gpu]["spans"] = int(gpu_render_stats[gpu]["spans"]) + 1
                gpu_render_stats[gpu]["video_sec"] = float(gpu_render_stats[gpu]["video_sec"]) + span_duration(span)
                gpu_render_stats[gpu]["wall_sec"] = float(gpu_render_stats[gpu]["wall_sec"]) + dt

        # Tail Idle Window: immediately run this GPU's assigned unrestored fast-bypass bucket
        # while remaining GPUs finish their final LTX render chunks.
        _run_unrestored_bucket_for_gpu(gpu)

    try:
        if pending_render or pending_unrestored:
            if pending_render:
                log.info(
                    "[Stage 2] Dispatching %d mosaic render spans + %d tail-overlapped unrestored spans across %d GPU(s) ...",
                    len(pending_render),
                    len(pending_unrestored),
                    num_gpus,
                )
            else:
                log.info(
                    "[Stage 2] Fast-encoding %d unrestored spans via NVENC across %d GPU(s) ...",
                    len(pending_unrestored),
                    num_gpus,
                )
            t_stage2_exec = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_gpus, thread_name_prefix="MultiGpuWorker") as pool:
                futs = [pool.submit(_gpu_worker_loop, g) for g in gpus]
                for fut in futs:
                    fut.result()
            stage2_wall_sec = time.perf_counter() - t_stage2_exec

            if pending_render:
                total_render_dur = sum(span_duration(s) for _, s in pending_render)
                max_gpu_render_wall = max(
                    (float(gpu_render_stats[g]["wall_sec"]) for g in gpus),
                    default=stage2_wall_sec,
                )
                for g in gpus:
                    st = gpu_render_stats[g]
                    phys = physical_cuda_visible_token(g)
                    gname = _gpu_device_label(g)
                    log.info(
                        "[Stage 2 修复明细] GPU=cuda:%d(phys:%s,%s) 分配段数=%d 处理视频时长=%s 单卡处理耗时=%s",
                        g,
                        phys,
                        gname,
                        int(st["spans"]),
                        format_duration(float(st["video_sec"])),
                        format_duration(float(st["wall_sec"])),
                    )
                log.info(
                    "[Stage 2 修复汇总] 通道B(LTX去码) 并行卡数=%d 总段数=%d 覆盖视频时长=%s 墙钟耗时=%s",
                    num_gpus,
                    len(pending_render),
                    format_duration(total_render_dur),
                    format_duration(max_gpu_render_wall),
                )
            if pending_unrestored:
                total_unrestored_dur = sum(span_duration(s) for _, s in pending_unrestored)
                max_ch_a_wall = max(ch_a_wall_times, default=0.0)
                log.info(
                    "[Stage 2 快编汇总] 通道A(NVENC快编) 并行卡数=%d 段数=%d 覆盖视频时长=%s 实际处理耗时=%s",
                    num_gpus,
                    len(pending_unrestored),
                    format_duration(total_unrestored_dur),
                    format_duration(max_ch_a_wall),
                )
    finally:
        for det in gpu_detectors.values():
            if hasattr(det, "close"):
                det.close()
        gpu_detectors.clear()

    if cancel_event.is_set():
        return False

    # Stage 3: FFmpeg parallel normalize, lossless concat, and final audio/subtitle/metadata remux
    log.info("[Stage 3] Normalizing and losslessly concatenating %d video fragments ...", len(spans))
    to_normalize = [
        (span_idx, raw_path(span_idx), norm_path(span_idx))
        for span_idx in range(len(spans))
        if not norm_path(span_idx).is_file()
    ]
    if to_normalize:
        norm_workers = min(8, len(to_normalize), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=norm_workers, thread_name_prefix="NormFrag") as pool:
            norm_futs = [
                pool.submit(normalize_fragment, src_nut, dst_norm, codec=config.codec)
                for _, src_nut, dst_norm in to_normalize
            ]
            for fut in norm_futs:
                fut.result()

    fragments: list[tuple[Path, float]] = [
        (norm_path(span_idx), span_duration(span))
        for span_idx, span in enumerate(spans)
    ]

    assembled = segments_dir / f"assembled{fragment_suffix}"
    concatenate_fragments(
        fragments,
        manifest=segments_dir / "fragments.ffconcat",
        destination=assembled,
        codec=config.codec,
    )
    mux_final_output(
        assembled,
        input_video,
        output_video,
        codec=config.codec,
    )

    process_sec = time.perf_counter() - t_stage2_start
    active_task = get_active_task_log()
    if active_task is not None:
        active_task.stats["process_sec"] = process_sec
        active_task.stats["output_path"] = str(output_video)
        active_task.stats["all_reused"] = all_reused
        if not all_reused and metadata.num_frames > 0:
            active_task.stats["frames"] = metadata.num_frames

    shutil.rmtree(resume_root, ignore_errors=True)
    log.info("=== Completed %s -> %s ===", input_video.name, output_video.name)
    return True


def _run_unrestored_worker_cli(task_file: Path) -> None:
    from types import SimpleNamespace
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import SpliceSpan, probe_keyframes

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    task = read_json(task_file)
    if not isinstance(task, dict):
        raise RuntimeError(f"Invalid unrestored task file: {task_file}")
    input_video = Path(task["input_video"])
    metadata = get_video_meta_data(str(input_video))
    index = probe_keyframes(input_video, metadata)
    cfg = SimpleNamespace(
        codec=task["codec"],
        encoder_settings=dict(task.get("encoder_settings") or {}),
        lut_path=task.get("lut_path"),
        sharpen_strength=float(task.get("sharpen_strength", 0.0)),
        batch_size=int(task.get("batch_size", 4)),
    )
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cancel_event = threading.Event()
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    spans_list = task.get("spans") or []
    for pos, item in enumerate(spans_list, 1):
        final_nut = Path(item["final_nut"])
        if final_nut.is_file():
            continue
        partial_nut = Path(item["partial_nut"])
        span = SpliceSpan("reencode", int(item["start_pts"]), int(item["end_pts"]))
        dur = float((span.end_pts - span.start_pts) * index.time_base)
        log.info(
            "[Stage 2][FastBypass phys:%s] Encoding unrestored span %d (%d/%d, %.1fs) ...",
            phys,
            int(item["idx"]) + 1,
            pos,
            len(spans_list),
            dur,
        )
        _encode_unrestored_span(
            cfg,
            input_video,
            metadata,
            index,
            span,
            partial_nut,
            final_nut,
            device=dev,
            cancel_event=cancel_event,
        )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--unrestored-worker":
        _run_unrestored_worker_cli(Path(sys.argv[2]))
    else:
        raise SystemExit(f"Unsupported arguments: {sys.argv[1:]}")
