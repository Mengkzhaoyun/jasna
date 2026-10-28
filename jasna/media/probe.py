"""Video stream metadata read through ffprobe."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

if TYPE_CHECKING:
    from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

logger = logging.getLogger(__name__)


class UnsupportedVideoError(Exception):
    pass


@dataclass
class VideoMetadata:
    video_file: str
    video_height: int
    video_width: int
    video_fps: float
    average_fps: float
    video_fps_exact: Fraction
    codec_name: str
    duration: float
    time_base: Fraction
    start_pts: int
    color_range: AvColorRange
    color_space: AvColorspace
    num_frames: int
    is_10bit: bool
    sample_aspect_ratio: Fraction = Fraction(1, 1)
    pixel_format: str = ""
    profile: str = ""
    field_order: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
    stereo_layout: str = ""
    spherical_projection: str = ""
    video_bitrate: int = 0
    stream_bitrate_bps: int | None = None
    stream_bitrate_source: str = "无法获取"
    container_bitrate_bps: int | None = None
    container_bitrate_source: str = "无法获取"
    file_size_bytes: int | None = None


def resolve_video_start_pts(stream_start_time: int | None, metadata_start_pts: int) -> int:
    if stream_start_time is not None:
        return int(stream_start_time)
    return metadata_start_pts

def _frame_count_from_container(path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(frame_count)


def is_stream_10bit(json_video_stream: dict) -> bool:
    bprs = json_video_stream.get('bits_per_raw_sample')
    if isinstance(bprs, (int, float)):
        return int(bprs) == 10
    if isinstance(bprs, str) and bprs.strip() == "10":
        return True
    pix_fmt = (json_video_stream.get('pix_fmt') or '').lower()
    ten_bit_markers = (
        'p10',
        'p010',
        'v210',
        'rgb10', 'bgr10', 'x2rgb10', 'x2bgr10', 'yuv10', 'gray10'
    )
    return any(marker in pix_fmt for marker in ten_bit_markers)

def parse_sample_aspect_ratio(json_video_stream: dict) -> Fraction:
    text = json_video_stream.get('sample_aspect_ratio') or ''
    num, sep, den = text.partition(':')
    if sep and num.isdigit() and den.isdigit() and int(num) > 0 and int(den) > 0:
        return Fraction(int(num), int(den))
    return Fraction(1, 1)


def parse_stream_and_container_bitrate(
    json_video_stream: dict, json_video_format: dict
) -> tuple[int | None, str, int | None, str]:
    """Return ``(stream_bps, stream_source, container_bps, container_source)`` without conflating stream vs container."""
    tags = json_video_stream.get("tags") or {}
    stream_candidates = (
        ("视频流bit_rate", json_video_stream.get("bit_rate")),
        ("视频流tag:BPS", tags.get("BPS")),
        ("视频流tag:BPS-eng", tags.get("BPS-eng")),
    )
    stream_bps: int | None = None
    stream_source = "无法获取"
    for label, candidate in stream_candidates:
        if candidate is None:
            continue
        try:
            value = int(float(candidate))
        except (TypeError, ValueError):
            continue
        if value > 0:
            stream_bps = value
            stream_source = label
            break

    container_bps: int | None = None
    container_source = "无法获取"
    fmt_br = json_video_format.get("bit_rate")
    if fmt_br is not None:
        try:
            value = int(float(fmt_br))
            if value > 0:
                container_bps = value
                container_source = "容器bit_rate"
        except (TypeError, ValueError):
            pass
    return stream_bps, stream_source, container_bps, container_source


def parse_video_bitrate(json_video_stream: dict, json_video_format: dict) -> int:
    """Video bitrate in bits/s, or 0 when no source reports one."""
    stream_bps, _, container_bps, _ = parse_stream_and_container_bitrate(
        json_video_stream, json_video_format
    )
    if stream_bps is not None:
        return stream_bps
    if container_bps is not None:
        return container_bps
    return 0


def parse_spatial_metadata(json_video_stream: dict) -> tuple[str, str]:
    stereo_layout = ""
    spherical_projection = ""
    for side_data in json_video_stream.get("side_data_list") or ():
        side_data_type = str(side_data.get("side_data_type") or "").lower()
        if side_data_type == "stereo 3d":
            stereo_layout = str(side_data.get("type") or "")
        elif side_data_type == "spherical mapping":
            spherical_projection = str(side_data.get("projection") or "")
    tags = json_video_stream.get("tags") or {}
    if not stereo_layout:
        stereo_layout = str(tags.get("stereo_mode") or "")
    if not spherical_projection:
        spherical_projection = str(tags.get("projection") or "")
    return stereo_layout, spherical_projection


def get_video_meta_data(path: str) -> VideoMetadata:
    from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
    ffprobe = resolve_executable("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-select_streams",
        "v",
        "-show_streams",
        "-show_format",
        path,
    ]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **subprocess_no_window_kwargs(),
    )
    out, err = p.communicate()
    if p.returncode != 0:
        stdout_text = (out or b"").decode(errors="replace")
        stderr_text = (err or b"").decode(errors="replace")
        logger.error(
            "ffprobe failed (exit code %s). stdout:\n%s\nstderr:\n%s",
            p.returncode,
            stdout_text,
            stderr_text,
        )
        raise RuntimeError(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    # avg_frame_rate is 0/0 when ffprobe cannot count the frames.
    avg_num, avg_den = (int(num) for num in json_video_stream['avg_frame_rate'].split("/"))
    average_fps = avg_num / avg_den if avg_den else float(avg_num)
    fps_exact = Fraction(json_video_stream['r_frame_rate'])
    fps = float(fps_exact)
    time_base = Fraction(json_video_stream['time_base'])
    start_pts = int(json_video_stream.get('start_pts') or 0)
    range_name = (json_video_stream.get("color_range") or "").lower()
    color_range = (
        AvColorRange.JPEG
        if range_name in {"pc", "jpeg", "full"}
        else AvColorRange.MPEG
    )
    color_space_name = (json_video_stream.get("color_space") or "").lower()
    if color_space_name in {"bt601", "bt470bg", "smpte170m"}:
        color_space = AvColorspace.ITU601
    elif color_space_name in {"bt2020", "bt2020nc", "bt2020_ncl"}:
        color_space = AvColorspace.BT2020
    else:
        color_space = AvColorspace.ITU709


    num_frames = int(json_video_stream.get('nb_frames', 0))
    if num_frames == 0:
        num_frames = _frame_count_from_container(path)
    is_10bit = is_stream_10bit(json_video_stream)
    stereo_layout, spherical_projection = parse_spatial_metadata(json_video_stream)
    stream_bps, stream_src, container_bps, container_src = parse_stream_and_container_bitrate(
        json_video_stream, json_video_format
    )
    file_size_bytes: int | None = None
    fmt_size = json_video_format.get("size")
    if fmt_size is not None:
        try:
            val = int(fmt_size)
            if val >= 0:
                file_size_bytes = val
        except (TypeError, ValueError):
            pass

    metadata = VideoMetadata(
        video_file=path,
        video_height=int(json_video_stream['height']),
        video_width=int(json_video_stream['width']),
        video_fps=fps,
        average_fps=average_fps,
        video_fps_exact=fps_exact,
        codec_name=json_video_stream['codec_name'],
        duration=float(json_video_stream.get('duration', json_video_format['duration'])),
        time_base=time_base,
        start_pts=start_pts,
        color_range=color_range,
        color_space=color_space,
        num_frames=num_frames,
        is_10bit=is_10bit,
        sample_aspect_ratio=parse_sample_aspect_ratio(json_video_stream),
        pixel_format=str(json_video_stream.get("pix_fmt") or ""),
        profile=str(json_video_stream.get("profile") or ""),
        field_order=str(json_video_stream.get("field_order") or ""),
        color_primaries=str(json_video_stream.get("color_primaries") or ""),
        color_transfer=str(json_video_stream.get("color_transfer") or ""),
        stereo_layout=stereo_layout,
        spherical_projection=spherical_projection,
        video_bitrate=parse_video_bitrate(json_video_stream, json_video_format),
        stream_bitrate_bps=stream_bps,
        stream_bitrate_source=stream_src,
        container_bitrate_bps=container_bps,
        container_bitrate_source=container_src,
        file_size_bytes=file_size_bytes,
    )
    return metadata
