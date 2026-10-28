"""Single-file batch runtime logging for Jasna video restoration tasks.

Implements the logging contract in ``vela/docs/runtime-log-review.md``:
- One log file per batch under ``.tmp/restoration-order/private/runtime-logs/<run-id>/jasna.log``
- Line format: ``YYYY-MM-DD HH:mm:ss Jasna <filename> [<index>/<total>|批次] <event> <message>``
- Captures Python stdout/stderr, logging handlers, and native/child-process fd 1/2 output
- Records input metadata & bitrate sources, effective encoder targets, Stage 1 analysis summary
  (analysis time, skipped non-mosaic/head-mosaic duration & percentage), Stage 2 per-GPU processing
  breakdown, and final verified output metadata.
"""

from __future__ import annotations

import argparse
import codecs
import io
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from jasna.media.probe import VideoMetadata
    from jasna.session_config import SessionConfig

_CAPTURE_LOCK = threading.RLock()
_ACTIVE_TASK_LOCK = threading.RLock()
_ACTIVE_TASK: TaskLog | None = None


def is_child_worker_process() -> bool:
    """Return True when running inside a spawned multi-GPU child worker subprocess."""
    main_pid = os.environ.get("JASNA_MAIN_PID", "").strip()
    if main_pid and main_pid.isdigit() and int(main_pid) != os.getpid():
        return True
    if os.environ.get("JASNA_WORKER_SUBPROCESS", "").strip() == "1":
        return True
    return False


def get_active_task_log() -> TaskLog | None:
    with _ACTIVE_TASK_LOCK:
        return _ACTIVE_TASK


@contextmanager
def _bind_active_task(task: TaskLog) -> Iterator[None]:
    global _ACTIVE_TASK
    with _ACTIVE_TASK_LOCK:
        prev = _ACTIVE_TASK
        _ACTIVE_TASK = task
    try:
        yield
    finally:
        with _ACTIVE_TASK_LOCK:
            _ACTIVE_TASK = prev


def format_bitrate(bps: int | float | None) -> str:
    """Display measured bit/s in decimal Mbps without turning unknown values into zero."""
    if bps is None or not math.isfinite(bps) or bps <= 0:
        return "无法获取"
    return f"{float(bps) / 1_000_000:.3f}Mbps"


def format_size(size_bytes: int | float | None) -> str:
    """Display byte counts in decimal MB / GB consistently with Mbps."""
    if size_bytes is None or not math.isfinite(size_bytes) or size_bytes < 0:
        return "未记录"
    val = float(size_bytes)
    if val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}GB"
    return f"{val / 1_000_000:.2f}MB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "未记录"
    centiseconds = round(float(seconds) * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds_text = f"{remainder / 100:.2f}".rstrip("0").rstrip(".")
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分")
    if remainder or not parts:
        parts.append(f"{seconds_text}秒")
    return "".join(parts)


def format_fps(fps: float | None) -> str:
    if fps is None or not math.isfinite(fps) or fps <= 0:
        return "无法获取"
    return f"{float(fps):.3f}".rstrip("0").rstrip(".") + "fps"


def _label(value: object) -> str:
    """Sanitize a filename or event label so it cannot forge a log line."""
    escaped: list[str] = []
    for char in str(value):
        if char == "\n":
            escaped.append(r"\n")
        elif char == "\r":
            escaped.append(r"\r")
        elif char == "\t":
            escaped.append(r"\t")
        elif ord(char) < 32 or 127 <= ord(char) < 160 or char in "\u2028\u2029":
            escaped.append(f"\\u{ord(char):04x}")
        else:
            escaped.append(char)
    return "".join(escaped)


def default_log_file_path(run_id: str | None = None) -> Path:
    base_dir = Path(
        os.environ.get("JASNA_RUNTIME_LOG_DIR", "").strip()
        or ".tmp/restoration-order/private/runtime-logs"
    )
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
    return base_dir / run_id / "jasna.log"


class RunLog:
    """Single-file batch log writer with independent console tee."""

    def __init__(
        self,
        log_file: Path | None = None,
        console: TextIO | None = None,
        *,
        append_existing: bool = False,
    ) -> None:
        if log_file is None:
            env_file = os.environ.get("JASNA_BATCH_LOG_FILE", "").strip()
            if env_file:
                log_file = Path(env_file)
                append_existing = True
            else:
                log_file = default_log_file_path()
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append_existing else "x"
        self._file = self.path.open(mode, encoding="utf-8", newline="\n", buffering=1)
        self._console = sys.stdout if console is None else console
        self._owns_console = False
        try:
            console_fd = self._console.fileno()
        except (AttributeError, io.UnsupportedOperation, OSError, ValueError):
            pass
        else:
            duplicate: int | None = None
            try:
                self._console.flush()
                duplicate = os.dup(console_fd)
                self._console = os.fdopen(
                    duplicate,
                    "w",
                    encoding=getattr(self._console, "encoding", None) or "utf-8",
                    errors=getattr(self._console, "errors", None) or "replace",
                    newline="\n",
                    buffering=1,
                )
                self._owns_console = True
            except BaseException:
                if duplicate is not None:
                    os.close(duplicate)
                self._file.close()
                raise
        self._lock = threading.RLock()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            finally:
                if self._owns_console:
                    try:
                        self._console.close()
                    except (BrokenPipeError, UnicodeEncodeError, OSError):
                        pass

    def emit(
        self,
        event: str,
        message: str,
        filename: str = "-",
        index: int = 0,
        total: int = 0,
    ) -> None:
        with self._lock:
            if self._file.closed:
                raise ValueError("Cannot write to a closed Jasna run log")
            task = f"{index}/{total}" if index else "批次"
            prefix = f"{datetime.now():%Y-%m-%d %H:%M:%S} Jasna {_label(filename)} [{task}] {_label(event)}"
            lines = str(message).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if len(lines) > 1 and not lines[-1]:
                lines.pop()
            for line in lines:
                record = f"{prefix} {line}\n"
                self._file.write(record)
                self._file.flush()
                try:
                    self._console.write(record)
                    self._console.flush()
                except (BrokenPipeError, UnicodeEncodeError, OSError):
                    pass

    def task(self, filename: str, index: int, total: int) -> TaskLog:
        if index < 1 or total < index:
            raise ValueError("Task index must be between 1 and total")
        return TaskLog(self, filename, index, total)


class TaskLog:
    def __init__(self, run: RunLog, filename: str, index: int, total: int) -> None:
        self._run = run
        self.filename = filename
        self.index = index
        self.total = total
        self.stats: dict[str, Any] = {}

    def emit(self, event: str, message: str) -> None:
        self._run.emit(event, message, self.filename, self.index, self.total)

    @contextmanager
    def capture(self, *, capture_native: bool = True) -> Iterator[None]:
        with _bind_active_task(self), capture_output(
            lambda line: self.emit("处理日志", line),
            capture_native_stderr=capture_native,
            capture_native_stdout=capture_native,
        ):
            yield


class _LineCapture(io.TextIOBase):
    """Buffer partial writes per thread and emit complete lines."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit
        self._buffers: dict[int, str] = {}
        self._lock = threading.RLock()

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        with self._lock:
            thread_id = threading.get_ident()
            buffered = self._buffers.get(thread_id, "") + text
            lines = buffered.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            self._buffers[thread_id] = lines.pop()
            for line in lines:
                if line:
                    self._emit(line)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            for thread_id, line in list(self._buffers.items()):
                if line:
                    self._buffers[thread_id] = ""
                    self._emit(line)


@contextmanager
def _capture_native_stream(stream_fd: int, emit: Callable[[str], None]) -> Iterator[None]:
    """Capture C-library / child-process output written directly to fd 1 or fd 2."""
    stream_name = "stdout" if stream_fd == 1 else "stderr"
    saved_fd: int | None = None
    read_fd: int | None = None
    write_fd: int | None = None
    saved_inheritable = True
    redirected = False
    reader: threading.Thread | None = None
    errors: list[BaseException] = []
    stop = threading.Event()

    def read_stream(fd: int) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        lines = _LineCapture(emit)
        try:
            with os.fdopen(fd, "rb", buffering=0) as source:
                while chunk := source.read(8192):
                    if stop.is_set():
                        return
                    if not errors:
                        try:
                            lines.write(decoder.decode(chunk))
                        except BaseException as exc:
                            errors.append(exc)
                if not errors and not stop.is_set():
                    lines.write(decoder.decode(b"", final=True))
                    lines.flush()
        except BaseException as exc:
            errors.append(exc)

    try:
        try:
            saved_fd = os.dup(stream_fd)
            saved_inheritable = os.get_inheritable(stream_fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, stream_fd)
            redirected = True
            os.close(write_fd)
            write_fd = None
        except OSError:
            if saved_fd is not None:
                os.close(saved_fd)
                saved_fd = None
            if read_fd is not None:
                os.close(read_fd)
                read_fd = None
            if write_fd is not None:
                os.close(write_fd)
                write_fd = None
            yield
            return

        reader = threading.Thread(
            target=read_stream,
            args=(read_fd,),
            name=f"jasna-native-{stream_name}",
            daemon=True,
        )
        reader.start()
        read_fd = None
        yield
    finally:
        pending_exception = sys.exc_info()[0] is not None
        try:
            if redirected and saved_fd is not None:
                os.dup2(saved_fd, stream_fd, inheritable=saved_inheritable)
        finally:
            if saved_fd is not None:
                os.close(saved_fd)
            if write_fd is not None:
                os.close(write_fd)
            if read_fd is not None:
                os.close(read_fd)
            if reader is not None and reader.ident is not None:
                reader.join(timeout=5.0)
                if reader.is_alive():
                    stop.set()
        if errors and not pending_exception:
            raise RuntimeError(f"Failed to capture native {stream_name}") from errors[0]


@contextmanager
def _redirect_logging_handlers(target_stream: TextIO) -> Iterator[None]:
    """Redirect existing StreamHandlers on the root logger so Python log records flow through _LineCapture."""
    root = logging.getLogger()
    swapped: list[tuple[logging.StreamHandler, Any]] = []
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            try:
                handler.flush()
                swapped.append((handler, handler.stream))
                handler.stream = target_stream
            except Exception:
                pass
    try:
        yield
    finally:
        for handler, orig_stream in swapped:
            try:
                handler.flush()
                handler.stream = orig_stream
            except Exception:
                pass


@contextmanager
def capture_output(
    emit: Callable[[str], None],
    *,
    capture_native_stderr: bool = True,
    capture_native_stdout: bool | None = None,
) -> Iterator[None]:
    """Capture Python stdout/stderr, root logging StreamHandlers, and native fd 1/2."""
    with _CAPTURE_LOCK:
        stdout = _LineCapture(emit)
        stderr = _LineCapture(emit)
        if capture_native_stdout is None:
            capture_native_stdout = capture_native_stderr
        native_stdout = _capture_native_stream(1, emit) if capture_native_stdout else nullcontext()
        native_stderr = _capture_native_stream(2, emit) if capture_native_stderr else nullcontext()
        try:
            with (
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                _redirect_logging_handlers(stderr),
                native_stdout,
                native_stderr,
            ):
                yield
        finally:
            stdout.flush()
            stderr.flush()


def sanitize_cli_args(argv: list[str]) -> str:
    """Redact license/activation secrets from recorded CLI arguments."""
    redacted: list[str] = []
    skip_next = False
    secret_flags = {"--license-code", "--activation-key"}
    for arg in argv:
        if skip_next:
            redacted.append("<REDACTED>")
            skip_next = False
            continue
        if arg in secret_flags:
            redacted.append(arg)
            skip_next = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in secret_flags):
            key = arg.split("=", 1)[0]
            redacted.append(f"{key}=<REDACTED>")
            continue
        redacted.append(arg)
    return " ".join(redacted) if redacted else "(默认参数)"


def format_input_summary(meta: VideoMetadata | None, source_path: Path | None = None) -> str:
    if meta is None:
        return "输入=无法获取 视频码率=无法获取 视频时长=无法获取"
    shape = (
        f"{meta.video_height}p({meta.video_width}x{meta.video_height})"
        if meta.video_width and meta.video_height
        else "无法获取"
    )
    duration = format_duration(meta.duration) if meta.duration > 0 else "无法获取"
    size_bytes = meta.file_size_bytes
    if size_bytes is None and source_path is not None and source_path.is_file():
        try:
            size_bytes = source_path.stat().st_size
        except OSError:
            pass
    container_bps = meta.container_bitrate_bps
    container_src = meta.container_bitrate_source
    if container_bps is None and size_bytes and meta.duration > 0:
        container_bps = int(size_bytes * 8 / meta.duration)
        container_src = "按文件大小估算"
    frames_str = str(meta.num_frames) if meta.num_frames and meta.num_frames > 0 else "未记录"
    return (
        f"输入={shape}@{format_fps(meta.video_fps)} "
        f"视频码率={format_bitrate(meta.stream_bitrate_bps)} 视频时长={duration} "
        f"总帧数={frames_str} "
        f"视频码率来源={meta.stream_bitrate_source} "
        f"文件总码率={format_bitrate(container_bps)} "
        f"文件总码率来源={container_src} "
        f"大小={format_size(size_bytes)}"
    )


def _parse_rate_setting_bps(val: object) -> int | None:
    if val is None:
        return None
    text = str(val).strip().lower()
    if not text or text == "0":
        return 0
    try:
        if text.endswith("m"):
            return int(float(text[:-1]) * 1_000_000)
        if text.endswith("k"):
            return int(float(text[:-1]) * 1_000)
        return int(float(text))
    except ValueError:
        return None


def format_target_summary(
    config: SessionConfig,
    meta: VideoMetadata | None,
    output: Path,
    *,
    multi_gpu_enabled: bool = True,
    head_filter_enabled: bool = True,
    gpu_count: int = 1,
) -> str:
    if meta is not None and meta.video_width and meta.video_height:
        shape = f"{meta.video_height}p({meta.video_width}x{meta.video_height})"
        out_fps = meta.video_fps / 2.0 if (config.retarget_high_fps and meta.video_fps > 50.0) else meta.video_fps
        fps_str = format_fps(out_fps)
    else:
        shape = "与输入一致"
        fps_str = "与输入一致"

    settings = dict(config.encoder_settings or {})
    rc = settings.get("rc", "未设置")
    cq = settings.get("cq", "未设置")
    gop = settings.get("g", "未设置")
    maxrate_bps = _parse_rate_setting_bps(settings.get("maxrate"))
    bitrate_bps = _parse_rate_setting_bps(settings.get("b") or settings.get("bitrate"))

    configured_target_env = os.environ.get("TARGET_BITRATE", "").strip()
    cfg_target_bps = _parse_rate_setting_bps(configured_target_env) if configured_target_env else maxrate_bps

    if cfg_target_bps and cfg_target_bps > 0:
        cfg_target_str = format_bitrate(cfg_target_bps)
    else:
        cfg_target_str = "未设置"

    if maxrate_bps == 0:
        maxrate_str = "无上限(maxrate=0,低码率保护生效)"
        eff_target_str = "未设置(保持源片画质不压缩)"
    elif maxrate_bps and maxrate_bps > 0:
        maxrate_str = format_bitrate(maxrate_bps)
        eff_target_str = format_bitrate(bitrate_bps) if (bitrate_bps and bitrate_bps > 0) else maxrate_str
    elif meta is not None and meta.video_bitrate > 0:
        # Jasna VideoEncoder derives a 2x input bitrate ceiling when maxrate is omitted
        derived_ceiling = meta.video_bitrate * 2
        maxrate_str = f"{format_bitrate(derived_ceiling)}(按输入码率2x自动推导)"
        eff_target_str = "未设置"
    else:
        maxrate_str = "未设置"
        eff_target_str = format_bitrate(bitrate_bps) if (bitrate_bps and bitrate_bps > 0) else "未设置"

    codec_upper = config.codec.upper()
    encoder_name = f"{config.codec}_nvenc" if str(config.device).startswith("cuda") else config.codec
    if config.restoration_model_name == "ltx":
        prec = "nvfp4" if config.ltx_fast else "int8"
        model_str = f"LTX({config.ltx_model}-{prec})"
    else:
        model_str = config.restoration_model_name

    return (
        f"{shape}@{fps_str} 输出={output.name} 容器={output.suffix.lstrip('.').upper() or 'MP4'} "
        f"配置目标视频码率={cfg_target_str} 生效目标视频码率={eff_target_str} "
        f"码率上限={maxrate_str} 编码={codec_upper}({encoder_name}) "
        f"码率控制={rc} CQ={cq} GOP={gop} 模型={model_str} "
        f"检测={config.detection_model_name} "
        f"人头马赛克剔除={'开启' if head_filter_enabled else '关闭'} "
        f"多卡管线={'开启(' + str(gpu_count) + '卡)' if multi_gpu_enabled else '关闭'}"
    )


def format_output_summary(
    output: Path,
    stats: dict[str, Any],
    started_at: float,
    task: TaskLog,
) -> str:
    from jasna.media.probe import get_video_meta_data

    actual_output = Path(stats.get("output_path") or output)
    if not actual_output.is_file() and actual_output.with_suffix(".mp4").is_file():
        actual_output = actual_output.with_suffix(".mp4")
    if not actual_output.is_file():
        raise FileNotFoundError(f"Final output file not found: {actual_output}")
    size = actual_output.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Final output file is empty: {actual_output}")

    out_meta: VideoMetadata | None = None
    try:
        out_meta = get_video_meta_data(str(actual_output))
    except Exception as exc:
        task.emit("处理日志", f"输出元数据无法获取: {exc}")

    video_bitrate = out_meta.stream_bitrate_bps if out_meta else None
    video_source = out_meta.stream_bitrate_source if out_meta else "无法获取"
    container_bitrate = out_meta.container_bitrate_bps if out_meta else None
    container_source = out_meta.container_bitrate_source if out_meta else "无法获取"
    if out_meta and container_bitrate is None and out_meta.duration > 0:
        container_bitrate = int(size * 8 / out_meta.duration)
        container_source = "按文件大小估算"

    task.emit(
        "处理日志",
        f"输出视频码率来源={video_source} 输出文件总码率来源={container_source} 大小来源=最终文件字节数",
    )

    resolution = (
        f"{out_meta.video_width}x{out_meta.video_height}"
        if out_meta and out_meta.video_width and out_meta.video_height
        else "无法获取"
    )
    elapsed = max(0.0, time.monotonic() - started_at)
    analysis_sec = stats.get("analysis_sec")
    process_sec = stats.get("process_sec")
    if isinstance(analysis_sec, (int, float)) and process_sec is None:
        process_sec = max(0.0, elapsed - float(analysis_sec))

    frames = stats.get("frames")
    if frames is None and out_meta is not None and out_meta.num_frames > 0:
        if stats.get("all_reused"):
            frames = 0
        else:
            frames = out_meta.num_frames

    if isinstance(frames, int) and frames >= 0 and elapsed > 0:
        speed = f"{frames / elapsed:.1f}fps"
        frames_str = str(frames)
    else:
        speed = "未记录"
        frames_str = "未记录"

    extra_timing = ""
    if isinstance(analysis_sec, (int, float)) and isinstance(process_sec, (int, float)):
        extra_timing = f"总分析时间={format_duration(float(analysis_sec))} 总处理时间={format_duration(float(process_sec))} "

    return (
        f"输出={actual_output} 分辨率={resolution} "
        f"帧率={format_fps(out_meta.video_fps if out_meta else None)} "
        f"视频码率={format_bitrate(video_bitrate)} "
        f"文件总码率={format_bitrate(container_bitrate)} "
        f"大小={format_size(size)} "
        f"{extra_timing}"
        f"处理时长={format_duration(elapsed)} "
        f"处理速度={speed} 处理帧数={frames_str}"
    )


def _cli_helper() -> int:
    """CLI helper for .drone/entrypoint.sh batch wrapper to share a single jasna.log across tasks."""
    parser = argparse.ArgumentParser(prog="python -m jasna.runtime_log")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-batch")
    p_init.add_argument("--total", type=int, required=True)
    p_init.add_argument("--failure-policy", default="继续下一视频")
    p_init.add_argument("--params", default="")

    p_skip = sub.add_parser("skip")
    p_skip.add_argument("--log-file", type=Path, required=True)
    p_skip.add_argument("--name", required=True)
    p_skip.add_argument("--index", type=int, required=True)
    p_skip.add_argument("--total", type=int, required=True)
    p_skip.add_argument("--reason", default="输出已存在")
    p_skip.add_argument("--output", required=True)

    p_end = sub.add_parser("end-batch")
    p_end.add_argument("--log-file", type=Path, required=True)
    p_end.add_argument("--total", type=int, required=True)
    p_end.add_argument("--success", type=int, required=True)
    p_end.add_argument("--failed", type=int, required=True)
    p_end.add_argument("--skipped", type=int, required=True)
    p_end.add_argument("--interrupted", type=int, default=0)
    p_end.add_argument("--elapsed", type=float, required=True)
    p_end.add_argument("--exit-code", type=int, required=True)

    args = parser.parse_args()
    if args.cmd == "init-batch":
        with RunLog() as run_log:
            tz = datetime.now().astimezone().strftime("UTC%z")
            run_log.emit(
                "批次开始",
                f"运行编号={run_log.path.parent.name} 任务数={args.total} 执行方式=串行 "
                f"失败策略={args.failure_policy} 时区={tz} 日志={run_log.path}",
            )
            run_log.emit(
                "参数记录",
                f"实际参数={args.params or '(批量扫描默认参数)'} "
                "输入/输出视频码率来源=视频流bit_rate或tag:BPS 文件总码率来源=容器bit_rate 大小来源=最终文件字节数",
            )
            # Print log path on a dedicated marker line if requested via env
            out_path_file = os.environ.get("JASNA_INIT_LOG_PATH_OUT", "").strip()
            if out_path_file:
                Path(out_path_file).write_text(str(run_log.path), encoding="utf-8")
        return 0

    if args.cmd == "skip":
        with RunLog(args.log_file, append_existing=True) as run_log:
            task = run_log.task(args.name, args.index, args.total)
            task.emit("任务跳过", f"原因={args.reason} 输出={args.output}")
        return 0

    if args.cmd == "end-batch":
        with RunLog(args.log_file, append_existing=True) as run_log:
            unprocessed = max(0, args.total - args.success - args.failed - args.skipped - args.interrupted)
            if args.interrupted > 0 or args.exit_code >= 128:
                status = "中断"
            elif args.failed > 0 or args.exit_code != 0:
                status = "存在失败"
            elif args.skipped > 0:
                status = "含跳过"
            else:
                status = "全部成功"
            parts = [
                f"状态={status}",
                f"总数={args.total}",
                f"成功={args.success}",
                f"失败={args.failed}",
                f"跳过={args.skipped}",
            ]
            if args.interrupted > 0:
                parts.append(f"中断={args.interrupted}")
            parts.append(f"未处理={unprocessed}")
            parts.append(f"总耗时={format_duration(args.elapsed)}")
            parts.append(f"进程退出码={args.exit_code}")
            run_log.emit("批次结束", " ".join(parts))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_helper())

