from __future__ import annotations

import io
import logging
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jasna.media.media_files import folder_output_path
from jasna.media.probe import VideoMetadata, parse_stream_and_container_bitrate
from jasna.multi_gpu import DEFAULT_MAX_INNER_GAP_SECONDS
from jasna.runtime_log import (
    RunLog,
    format_bitrate,
    format_duration,
    format_fps,
    format_input_summary,
    format_output_summary,
    format_size,
    format_target_summary,
    get_active_task_log,
    sanitize_cli_args,
)


def _dummy_meta(
    *,
    width: int = 854,
    height: int = 480,
    fps: float = 59.94,
    duration: float = 8432.17,
    num_frames: int = 505425,
    stream_bps: int | None = 2_003_000,
    stream_src: str = "视频流tag:BPS",
    container_bps: int | None = 2_131_000,
    container_src: str = "容器bit_rate",
    size_bytes: int | None = 2_245_988_513,
) -> VideoMetadata:
    return VideoMetadata(
        video_file="sample.mkv",
        video_height=height,
        video_width=width,
        video_fps=fps,
        average_fps=fps,
        video_fps_exact=Fraction(60000, 1001),
        codec_name="h264",
        duration=duration,
        time_base=Fraction(1, 1000),
        start_pts=0,
        color_range=None,  # type: ignore[arg-type]
        color_space=None,  # type: ignore[arg-type]
        num_frames=num_frames,
        is_10bit=False,
        video_bitrate=stream_bps or container_bps or 0,
        stream_bitrate_bps=stream_bps,
        stream_bitrate_source=stream_src,
        container_bitrate_bps=container_bps,
        container_bitrate_source=container_src,
        file_size_bytes=size_bytes,
    )


class TestJasnaRuntimeLog(unittest.TestCase):
    def test_format_helpers(self) -> None:
        self.assertEqual(format_bitrate(None), "无法获取")
        self.assertEqual(format_bitrate(0), "无法获取")
        self.assertEqual(format_bitrate(2_744_707), "2.745Mbps")
        self.assertEqual(format_size(3_045_216_940), "3.05GB")
        self.assertEqual(format_size(38_700_000), "38.70MB")
        self.assertEqual(format_duration(8432.17), "2小时20分32.17秒")
        self.assertEqual(format_duration(84.19), "1分24.19秒")
        self.assertEqual(format_fps(59.94005994), "59.94fps")
        self.assertEqual(
            sanitize_cli_args(["--input", "a.mkv", "--license-code", "JL1-SECRET", "--codec", "hevc"]),
            "--input a.mkv --license-code <REDACTED> --codec hevc",
        )

    def test_stream_and_container_bitrate_kept_separate(self) -> None:
        stream_bps, stream_src, cont_bps, cont_src = parse_stream_and_container_bitrate(
            {"tags": {}},
            {"bit_rate": "2131000"},
        )
        self.assertIsNone(stream_bps)
        self.assertEqual(stream_src, "无法获取")
        self.assertEqual(cont_bps, 2_131_000)
        self.assertEqual(cont_src, "容器bit_rate")

        stream_bps2, stream_src2, cont_bps2, cont_src2 = parse_stream_and_container_bitrate(
            {"tags": {"BPS": "2003000"}},
            {"bit_rate": "2131000"},
        )
        self.assertEqual(stream_bps2, 2_003_000)
        self.assertEqual(stream_src2, "视频流tag:BPS")
        self.assertEqual(cont_bps2, 2_131_000)
        self.assertEqual(cont_src2, "容器bit_rate")

    def test_run_log_and_task_capture_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run-01" / "jasna.log"
            console = io.StringIO()
            with RunLog(log_path, console=console) as run_log:
                run_log.emit("批次开始", "运行编号=run-01 任务数=1 时区=UTC+08:00")
                task = run_log.task("sample.mkv", 1, 1)
                meta = _dummy_meta()
                task.emit("任务开始", format_input_summary(meta))
                with task.capture(capture_native=False):
                    self.assertIs(get_active_task_log(), task)
                    print("LTX 1/4 scan: 100%\rLTX 2/4 encode: 50%")
                    sys.stderr.write("[Stage 1 分析汇总] 分析时间=1分24.19秒\n")
                    logging.getLogger("jasna.multi_gpu").info("[Stage 2 修复汇总] 通道B(LTX去码) 墙钟耗时=59分45秒")
                    task.stats["analysis_sec"] = 84.19
                    task.stats["process_sec"] = 3650.0
                    task.stats["frames"] = 505425
                self.assertIsNone(get_active_task_log())

            content = log_path.read_text(encoding="utf-8")
            self.assertEqual(content, console.getvalue())
            self.assertIn("Jasna - [批次] 批次开始 运行编号=run-01", content)
            self.assertIn(
                "Jasna sample.mkv [1/1] 任务开始 输入=480p(854x480)@59.94fps 视频码率=2.003Mbps 视频时长=2小时20分32.17秒 总帧数=505425 视频码率来源=视频流tag:BPS 文件总码率=2.131Mbps 文件总码率来源=容器bit_rate 大小=2.25GB",
                content,
            )
            self.assertIn("Jasna sample.mkv [1/1] 处理日志 LTX 1/4 scan: 100%", content)
            self.assertIn("Jasna sample.mkv [1/1] 处理日志 LTX 2/4 encode: 50%", content)
            self.assertIn("Jasna sample.mkv [1/1] 处理日志 [Stage 1 分析汇总] 分析时间=1分24.19秒", content)

    def test_target_and_output_summaries(self) -> None:
        meta = _dummy_meta()
        cfg = SimpleNamespace(
            device="cuda:0",
            codec="hevc",
            encoder_settings={"rc": "vbr", "maxrate": "0", "bufsize": "0", "cq": 20, "g": 60},
            restoration_model_name="ltx",
            ltx_model="distilled2",
            ltx_fast=False,
            detection_model_name="rfdetr-v6-large",
            retarget_high_fps=False,
        )
        target_line = format_target_summary(
            cfg,  # type: ignore[arg-type]
            meta,
            Path("sample-U.mp4"),
            multi_gpu_enabled=True,
            head_filter_enabled=True,
            gpu_count=4,
        )
        self.assertIn("输出=sample-U.mp4 容器=MP4", target_line)
        self.assertIn("码率上限=无上限(maxrate=0,低码率保护生效)", target_line)
        self.assertIn("编码=HEVC(hevc_nvenc)", target_line)
        self.assertIn("多卡管线=开启(4卡)", target_line)

        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "sample-U.mp4"
            out_file.write_bytes(b"x" * 3_045_216)
            log_path = Path(tmp) / "jasna.log"
            out_meta = _dummy_meta(
                stream_bps=2_744_707,
                stream_src="视频流bit_rate",
                container_bps=2_889_141,
                container_src="容器bit_rate",
                size_bytes=3_045_216,
            )
            with RunLog(log_path, console=io.StringIO()) as run_log:
                task = run_log.task("sample.mkv", 1, 1)
                task.stats = {
                    "analysis_sec": 84.19,
                    "process_sec": 3710.0,
                    "frames": 505425,
                    "output_path": str(out_file),
                }
                with patch("jasna.media.probe.get_video_meta_data", return_value=out_meta):
                    summary = format_output_summary(out_file, task.stats, 0.0, task)
                task.emit("处理完毕", summary)

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("输出视频码率来源=视频流bit_rate 输出文件总码率来源=容器bit_rate 大小来源=最终文件字节数", text)
            self.assertIn("分辨率=854x480 帧率=59.94fps 视频码率=2.745Mbps 文件总码率=2.889Mbps", summary)
            self.assertIn("总分析时间=1分24.19秒 总处理时间=1小时1分50秒", summary)
            self.assertIn("处理帧数=505425", summary)

    def test_folder_output_path_defaults_video_to_mp4(self) -> None:
        self.assertEqual(
            folder_output_path("/out", "/in/movie.mkv"),
            Path("/out/movie_out.mp4"),
        )
        self.assertEqual(
            folder_output_path("/out", "/in/poster.png"),
            Path("/out/poster_out.png"),
        )
        self.assertEqual(DEFAULT_MAX_INNER_GAP_SECONDS, 35.0)


if __name__ == "__main__":
    unittest.main()
