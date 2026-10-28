#!/usr/bin/env python3
"""jasna.protected_io —— 受保护模型载荷的离线还原与巡查入口。

入口
----
由 ``jasna/__main__.py`` 的隐藏分支转发（与 ``--compile-engines`` 同一层）::

    jasna --export-models <子命令> [选项]

该分支位于 ``initialize_runtime()`` 之后，因此受保护运行时与许可闸门已就绪。

为什么只能在冻结产物里跑
------------------------
``jasna.model_files.open_tensors`` 的分派取决于 ``jasna._frozen.is_frozen()``：
源码态走 ``safetensors.safe_open`` 明文分支，冻结态才走
``jasna.protection.protected_tensors.open_ltx_tensors``。
而 ``protected_tensors`` 只存在于发布产物中，源码树内只有公开桩
（``jasna/protection/__init__.py`` 里的 ``_ProtectedModel`` 全抛 ProtectionError）。

因此本模块只在冻结产物中可用；在源码树里 ``license`` 之外的子命令会明确报错。

不重实现密码学
--------------
解密封装全部交给产品自身的接口：

* ``*.safetensors.enc`` → ``open_ltx_tensors``（与 ``safe_open`` 同构的 reader）
* ``*.onnx.enc``        → ``protected_model.decrypt_model_bytes``
* ``*.engine.enc``      → ``protected_model.decrypt_engine_bytes``
* ``*.ckpt.enc``        → ``protected_model.decrypt_model_to_buffer``

本模块只负责调度、判定许可与写盘。

子命令
------
``license``          仅判定许可与保护模块可用性（不写文件，先用它验证链路）
``inspect``          打开目录内每个 ``*.enc``，报告张量数与布局（不写明文）
``materialize``      把单个 ``.enc`` 流式导出为明文（默认拒绝，需显式确认）
``materialize-dir``  批量导出目录内所有 ``*.enc``，按后缀分派到产品自身的解密接口

``materialize*`` 会把受保护权重落成**无保护明文**，等于放弃产品的防提取边界。
它面向权利人做离线迁移/取证，**不要随客户发行包交付**。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

TOOL = "jasna --export-models"
VERSION = "1.0.0"

#: safetensors dtype -> 每元素字节数。
DTYPE_BYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

#: safetensors dtype -> 同宽度的整数视图，用于零拷贝取字节。
DTYPE_VIEW: Dict[str, str] = {
    "BOOL": "uint8",
    "U8": "uint8",
    "I8": "uint8",
    "I16": "int16",
    "BF16": "int16",
    "F16": "int16",
    "I32": "int32",
    "F32": "int32",
    "I64": "int64",
    "F64": "int64",
    "F8_E4M3": "uint8",
    "F8_E5M2": "uint8",
}


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str, code: int = 1) -> int:
    print(f"错误: {message}", file=sys.stderr, flush=True)
    return code


# ---------------------------------------------------------------------------
# 后端：优先 torch（冻结产物必然有），缺 torch 时退化为 numpy 以便单测
# ---------------------------------------------------------------------------


class Backend:
    """分配目标缓冲并把缓冲转成字节视图。

    冻结产物里一定有 torch；缺 torch 时退化为 numpy，仅为让写出逻辑可单测。
    """

    def __init__(self) -> None:
        try:
            import torch

            self.torch = torch
            self.np = None
            self.kind = "torch"
        except ImportError:
            import numpy

            self.torch = None
            self.np = numpy
            self.kind = "numpy"

    _TORCH_NAMES = {
        "BOOL": "bool",
        "U8": "uint8",
        "I8": "int8",
        "I16": "int16",
        "I32": "int32",
        "I64": "int64",
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
        "F64": "float64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
    }

    _NUMPY_NAMES = {
        "BOOL": "bool_",
        "U8": "uint8",
        "I8": "int8",
        "I16": "int16",
        "I32": "int32",
        "I64": "int64",
        "F16": "float16",
        "BF16": "uint16",  # 无 bf16，用同宽整数承载原始位
        "F32": "float32",
        "F64": "float64",
        "F8_E4M3": "uint8",
        "F8_E5M2": "uint8",
    }

    def torch_dtype_for(self, st_dtype: str):
        if self.torch is None:
            return None
        return getattr(self.torch, self._TORCH_NAMES[st_dtype])

    def alloc(self, shape: Sequence[int], st_dtype: str):
        if self.kind == "torch":
            return self.torch.empty(tuple(shape), dtype=self.torch_dtype_for(st_dtype))
        return self.np.zeros(tuple(shape), dtype=getattr(self.np, self._NUMPY_NAMES[st_dtype]))

    def raw_view(self, buffer, st_dtype: str):
        """返回同宽度整数的一维视图，零拷贝。"""
        view = DTYPE_VIEW[st_dtype]
        if self.kind == "torch":
            target = getattr(self.torch, view)
            return buffer.view(target).numpy().reshape(-1)
        return buffer.view(getattr(self.np, view)).reshape(-1)


# ---------------------------------------------------------------------------
# 许可与保护可用性
# ---------------------------------------------------------------------------


@dataclass
class LicenseState:
    frozen: bool
    protection_available: bool
    licensed: bool
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frozen": self.frozen,
            "protection_available": self.protection_available,
            "licensed": self.licensed,
            "detail": self.detail,
        }


def license_state() -> LicenseState:
    """判定冻结状态、保护模块可用性与许可状态（只读）。"""
    try:
        from jasna._frozen import is_frozen
    except Exception as exc:  # noqa: BLE001
        return LicenseState(False, False, False, f"无法导入 jasna._frozen: {exc}")

    frozen = bool(is_frozen())

    try:
        from jasna.license_api import protection_available

        available = bool(protection_available())
    except Exception as exc:  # noqa: BLE001
        return LicenseState(frozen, False, False, f"license_api 不可用: {exc}")

    if not available:
        return LicenseState(
            frozen,
            False,
            False,
            "保护模块不可用：源码态且缺少 jasna.protection.protected_tensors。",
        )

    try:
        from jasna.protection.protected_tensors import check_ltx_license

        check_ltx_license()
    except Exception as exc:  # noqa: BLE001
        return LicenseState(frozen, True, False, f"许可校验未通过: {type(exc).__name__}: {exc}")

    return LicenseState(frozen, True, True, "许可校验通过")


def open_reader(path: Path):
    """用产品自身分派打开 reader（明文或 .enc）。"""
    from jasna.model_files import open_tensors

    return open_tensors(path)


# ---------------------------------------------------------------------------
# reader 巡查
# ---------------------------------------------------------------------------


def reader_specs(reader) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """把 reader.tensor_specs() 归一到 safetensors dtype 字符串。"""
    raw = reader.tensor_specs()
    try:
        import torch

        to_string = {
            torch.bool: "BOOL",
            torch.uint8: "U8",
            torch.int8: "I8",
            torch.int16: "I16",
            torch.int32: "I32",
            torch.int64: "I64",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.float32: "F32",
            torch.float64: "F64",
        }
        for name in ("float8_e4m3fn", "float8_e5m2"):
            attr = getattr(torch, name, None)
            if attr is not None:
                to_string[attr] = "F8_E4M3" if name.endswith("e4m3fn") else "F8_E5M2"
    except ImportError:
        to_string = {}

    result: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for name, value in raw.items():
        dtype, shape = value
        if isinstance(dtype, str) and dtype.upper() in DTYPE_BYTES:
            key = dtype.upper()
        elif dtype in to_string:
            key = to_string[dtype]
        else:
            raise ValueError(f"{name}: 无法识别的 dtype {dtype!r}")
        result[name] = (key, tuple(int(dim) for dim in shape))
    return result


def plan_layout(
    specs: Dict[str, Tuple[str, Tuple[int, ...]]], order: Sequence[str]
) -> Tuple[Dict[str, Any], int]:
    """计算 safetensors 头部与数据区总长度。"""
    header: Dict[str, Any] = {}
    offset = 0
    for name in order:
        dtype, shape = specs[name]
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    return header, offset


def encode_header(header: Dict[str, Any]) -> bytes:
    payload = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    pad = (-len(payload)) % 8
    return payload + b" " * pad


def inspect_file(path: Path) -> Dict[str, Any]:
    handle = open_reader(path)
    with handle as reader:
        specs = reader_specs(reader)
        metadata = dict(reader.metadata() or {})
        try:
            order = list(reader.offset_keys())
        except Exception:  # noqa: BLE001
            order = list(specs)
        header, data_bytes = plan_layout(specs, order)
        total = len(encode_header(header)) + 8 + data_bytes
    result = {
        "file": str(path),
        "tensors": len(specs),
        "data_bytes": data_bytes,
        "projected_safetensors_bytes": total,
        "file_bytes": path.stat().st_size,
        "overhead_bytes": path.stat().st_size - data_bytes,
        "metadata_keys": sorted(metadata),
        "dtypes": sorted({dtype for dtype, _ in specs.values()}),
        "sample": [
            {"name": name, "dtype": specs[name][0], "shape": list(specs[name][1])}
            for name in order[:5]
        ],
    }
    return result


# ---------------------------------------------------------------------------
# 流式导出
# ---------------------------------------------------------------------------


def materialize(
    source: Path,
    destination: Path,
    backend: Backend,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """把 reader 逐张量流式写成 safetensors，内存占用约等于单个最大张量。"""
    handle = open_reader(source)
    started = time.time()
    with handle as reader:
        specs = reader_specs(reader)
        metadata = {str(k): str(v) for k, v in (reader.metadata() or {}).items()}
        try:
            order = list(reader.offset_keys())
        except Exception:  # noqa: BLE001
            order = list(specs)
        missing = [name for name in order if name not in specs]
        if missing:
            raise ValueError(f"offset_keys 含未知张量: {missing[:3]}")

        header, data_bytes = plan_layout(specs, order)
        if metadata:
            header["__metadata__"] = metadata
        encoded = encode_header(header)

        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with destination.open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for index, name in enumerate(order, 1):
                dtype, shape = specs[name]
                buffer = backend.alloc(shape, dtype)
                reader.get_tensor_into(name, buffer)
                view = backend.raw_view(buffer, dtype)
                expected = DTYPE_BYTES[dtype] * _count(shape)
                actual = int(view.nbytes)
                if actual != expected:
                    raise ValueError(
                        f"{name}: 写入 {actual} 字节，头部声明 {expected} 字节"
                    )
                out.write(memoryview(view))
                written += actual
                if progress is not None:
                    progress(name, index, len(order))
                del buffer, view

        expected_total = 8 + len(encoded) + data_bytes
        actual_total = destination.stat().st_size
        if actual_total != expected_total:
            raise ValueError(
                f"输出尺寸不符: {actual_total} != {expected_total}（剩余张量可能未写全）"
            )

    return {
        "source": str(source),
        "destination": str(destination),
        "tensors": len(order),
        "data_bytes": data_bytes,
        "written_bytes": written,
        "file_bytes": actual_total,
        "seconds": round(time.time() - started, 3),
    }


def _count(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def cmd_license(args: argparse.Namespace) -> int:
    state = license_state()
    log(json.dumps(state.as_dict(), ensure_ascii=False, indent=2))
    return 0 if state.licensed else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    state = license_state()
    if not state.licensed:
        return fail(state.detail)

    directory = Path(args.root).resolve()
    targets = sorted(directory.glob("*.enc")) if not args.file else [Path(args.file)]
    if not targets:
        return fail(f"{directory} 内没有 .enc 文件", 2)

    results = []
    problems = 0
    for path in targets:
        try:
            info = inspect_file(path)
        except Exception as exc:  # noqa: BLE001
            problems += 1
            results.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(info)
        log(
            f"[OK] {path.name}: {info['tensors']} 张量, {info['data_bytes'] / 1e9:.2f} GB 数据, "
            f"dtypes={','.join(info['dtypes'])}, metadata={','.join(info['metadata_keys']) or '-'}"
        )

    if args.report:
        Path(args.report).write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        log(f"报告已写入: {args.report}")
    return 0 if problems == 0 else 1


def _protected_model():
    """私有树里的 ``jasna.protection.protected_model`` 门面。"""
    from jasna.protection import protected_model

    return protected_model


#: 由文件名推断模型 id 时剥掉的载荷后缀（``unet-4x.onnx.enc`` -> ``unet-4x``）。
PAYLOAD_SUFFIXES = (".safetensors", ".onnx", ".engine", ".ckpt", ".pt", ".bin", ".trt")

#: 文件名与产品内部 MODEL_ID 不一致的场合，按产品常量显式映射：
#:   UNET4X_MODEL_ID = "unet-4x"        (unet4x_secondary_restorer.py:23)
#:   Sd15InpaintRestorer.MODEL_ID = "sd-15-jav"  (sd15_inpaint_restorer.py:54)
MODEL_ID_HINTS = {
    "unet-4x.onnx.enc": "unet-4x",
    "sd15-200000.ckpt.enc": "sd-15-jav",
}


def payload_model_id(name: str) -> str:
    if name in MODEL_ID_HINTS:
        return MODEL_ID_HINTS[name]
    stem = name[: -len(".enc")] if name.endswith(".enc") else name
    for suffix in PAYLOAD_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_tensor_payload(name: str) -> bool:
    return name.endswith(".safetensors.enc")


def materialize_payload(
    source: Path,
    destination: Path,
    model_id: str,
    max_bytes: int,
) -> Dict[str, Any]:
    """还原非张量载荷。按产品源码里的三条调用路径分派：

    * ``.onnx.enc``   → ``decrypt_model_bytes(model_id, Path)``
      （``unet4x_secondary_restorer.py:60``）
    * ``.engine.enc`` → ``decrypt_engine_bytes(model_id, bytes)``
      （``engine_compiler.py:107``、``unet4x_secondary_restorer.py:104``）
    * ``.ckpt.enc``   → ``decrypt_model_to_buffer(model_id, Path)``
      （``sd15_inpaint_restorer.py:44``）
    * 其他           → 退回 ``decrypt_model_bytes``

    注意三者的入参形态不同：两个收 ``Path``，一个收 ``bytes``。
    """
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"文件 {size} B 超过 --max-bytes {max_bytes}，拒绝整读")
    started = time.time()
    model = _protected_model()
    name = source.name

    def need(attr: str):
        func = getattr(model, attr, None)
        if func is None:
            raise RuntimeError(f"protected_model.{attr} 不存在：该私有构建未暴露该接口")
        return func

    if name.endswith(".engine.enc"):
        plain = need("decrypt_engine_bytes")(model_id, source.read_bytes())
        kind = "trt-engine"
    elif name.endswith(".ckpt.enc"):
        plain = need("decrypt_model_to_buffer")(model_id, source)
        kind = "checkpoint"
    elif name.endswith(".onnx.enc"):
        plain = need("decrypt_model_bytes")(model_id, source)
        kind = "onnx-model"
    else:
        plain = need("decrypt_model_bytes")(model_id, source)
        kind = "model-bytes"

    if not isinstance(plain, (bytes, bytearray, memoryview)):
        tobytes = getattr(plain, "tobytes", None)
        if tobytes is None:
            raise RuntimeError(f"{kind} 解密返回 {type(plain).__name__}，无法转为字节")
        plain = tobytes()
    plain = bytes(plain)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plain)
    return {
        "source": str(source),
        "destination": str(destination),
        "model_id": model_id,
        "kind": kind,
        "ciphertext_bytes": size,
        "plaintext_bytes": len(plain),
        "seconds": round(time.time() - started, 3),
    }


def cmd_materialize_dir(args: argparse.Namespace) -> int:
    """把一个目录下所有 ``*.safetensors.enc`` 批量还原为明文 safetensors。"""
    if not args.confirm_unprotected:
        return fail(
            "批量还原会把全部受保护权重写成无保护明文。确认后请加 "
            "--confirm-unprotected-output；输出目录应为 0700 且不与他人共享。",
            3,
        )

    state = license_state()
    if not state.licensed:
        return fail(state.detail)

    models = Path(args.models).resolve()
    if not models.is_dir():
        return fail(f"模型目录不存在: {models}")
    candidates = sorted(models.glob("*.enc"))
    if args.tensors_only:
        candidates = [p for p in candidates if is_tensor_payload(p.name)]
    if not candidates:
        return fail(f"{models} 下没有可还原的 *.enc", 2)
    if args.only:
        candidates = [p for p in candidates if args.only in p.name]
        if not candidates:
            return fail(f"没有匹配 --only {args.only!r} 的文件", 2)
    targets = candidates

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.allow_shared_out:
        mode = out_dir.stat().st_mode & 0o777
        if mode & 0o077:
            return fail(
                f"输出目录权限 {mode:o} 过宽：明文权重不应被同机其他用户读取。"
                "请 chmod 700 后重试，或显式加 --allow-shared-out。",
                3,
            )

    backend = Backend()
    log(
        f"后端: {backend.kind}；源目录: {models}；输出: {out_dir}；文件数: {len(targets)}"
    )
    tensor_count = sum(1 for p in targets if is_tensor_payload(p.name))
    log(f"其中张量载荷 {tensor_count} 个，字节载荷 {len(targets) - tensor_count} 个")

    manifest: List[Dict[str, Any]] = []
    failed: List[str] = []
    for index, source in enumerate(targets, 1):
        destination = out_dir / source.name[: -len(".enc")]
        log(f"[{index}/{len(targets)}] {source.name} -> {destination.name}")
        if destination.exists() and not args.overwrite:
            failed.append(f"{source.name}: 目标已存在")
            log("    跳过：目标已存在（--overwrite 覆盖）")
            continue
        try:
            if is_tensor_payload(source.name):
                summary = materialize(source, destination, backend)
                entry = {
                    "source": source.name,
                    "output": destination.name,
                    "kind": "safetensors-tensors",
                    "tensors": summary["tensors"],
                    "data_bytes": summary["data_bytes"],
                    "file_bytes": summary["file_bytes"],
                }
            else:
                model_id = args.model_id or payload_model_id(source.name)
                summary = materialize_payload(
                    source, destination, model_id, args.max_bytes
                )
                entry = {
                    "source": source.name,
                    "output": destination.name,
                    "kind": summary["kind"],
                    "model_id": summary["model_id"],
                    "ciphertext_bytes": summary["ciphertext_bytes"],
                    "file_bytes": summary["plaintext_bytes"],
                }
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{source.name}: {type(exc).__name__}: {exc}")
            log(f"    失败：{type(exc).__name__}: {exc}")
            continue
        entry["sha256"] = _sha256_file(destination)
        manifest.append(entry)
        log(f"    完成（{entry['kind']}）：{entry['file_bytes'] / 1e9:.2f} GB")

    report = {
        "schema": "jasna.model.export/1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_dir": str(models),
        "output_dir": str(out_dir),
        "exported": manifest,
        "failed": failed,
    }
    report_path = out_dir / "export_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"清单: {report_path}")
    log(f"成功 {len(manifest)} 个，失败 {len(failed)} 个")
    return 0 if not failed else 1


def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_materialize(args: argparse.Namespace) -> int:
    if not args.confirm_unprotected:
        return fail(
            "该操作会把受保护权重写成无保护明文，等同放弃防提取边界。"
            "确认后请加 --confirm-unprotected-output；不要随客户发行包交付。",
            3,
        )

    state = license_state()
    if not state.licensed:
        return fail(state.detail)

    source = Path(args.file).resolve()
    if not source.is_file():
        return fail(f"文件不存在: {source}")
    destination = (
        Path(args.out).resolve()
        if args.out
        else source.with_name(
            source.name[: -len(".enc")] if source.name.endswith(".enc") else source.name + ".plain"
        )
    )
    if destination.exists() and not args.overwrite:
        return fail(f"目标已存在: {destination}（加 --overwrite 覆盖）", 3)

    backend = Backend()
    log(f"后端: {backend.kind}；源: {source.name}；目标: {destination.name}")

    def progress(name: str, index: int, total: int) -> None:
        if index % 25 == 0 or index == total:
            log(f"  [{index}/{total}] {name}")

    try:
        summary = materialize(source, destination, backend, progress)
    except Exception as exc:  # noqa: BLE001
        return fail(f"导出失败: {type(exc).__name__}: {exc}")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("完成。注意：输出文件不含任何保护，分发前请评估。")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="在冻结发行版进程内处理 .enc 模型文件（需有效许可）。",
    )
    parser.add_argument("--version", action="version", version=f"{TOOL} {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_license = sub.add_parser("license", help="判定冻结状态与许可")
    p_license.set_defaults(func=cmd_license)

    p_inspect = sub.add_parser("inspect", help="打开 .enc 并报告布局（不写明文）")
    p_inspect.add_argument("--root", default=".")
    p_inspect.add_argument("--file", help="只检查单个文件")
    p_inspect.add_argument("--report", help="JSON 报告路径")
    p_inspect.set_defaults(func=cmd_inspect)

    p_mat = sub.add_parser("materialize", help="流式导出为明文 safetensors（权利人专用）")
    p_mat.add_argument("--file", required=True, help="源 .enc 文件")
    p_mat.add_argument("--out", help="输出路径；缺省为去掉 .enc 的同名文件")
    p_mat.add_argument("--overwrite", action="store_true")
    p_mat.add_argument(
        "--confirm-unprotected-output",
        dest="confirm_unprotected",
        action="store_true",
        help="确认输出为无保护明文（必需）",
    )
    p_mat.set_defaults(func=cmd_materialize)

    p_dir = sub.add_parser("materialize-dir", help="批量还原目录内所有 *.enc")
    p_dir.add_argument("--models", required=True, help="含 *.enc 的目录")
    p_dir.add_argument("--out", required=True, help="明文输出目录（应为 0700）")
    p_dir.add_argument("--only", help="只处理文件名包含该子串的文件")
    p_dir.add_argument("--overwrite", action="store_true")
    p_dir.add_argument(
        "--tensors-only",
        action="store_true",
        help="只处理 *.safetensors.enc，跳过引擎等字节载荷",
    )
    p_dir.add_argument(
        "--model-id",
        help="字节载荷的模型 id（默认由文件名推断，如 unet-4x.onnx.enc -> unet-4x）",
    )
    p_dir.add_argument(
        "--max-bytes",
        type=int,
        default=2 << 30,
        help="字节载荷整读上限，默认 2 GiB",
    )
    p_dir.add_argument(
        "--allow-shared-out",
        action="store_true",
        help="允许输出目录为组/其他用户可读（不推荐）",
    )
    p_dir.add_argument(
        "--confirm-unprotected-output",
        dest="confirm_unprotected",
        action="store_true",
        help="确认输出为无保护明文（必需）",
    )
    p_dir.set_defaults(func=cmd_materialize_dir)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
