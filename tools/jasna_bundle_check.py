#!/usr/bin/env python3
"""jasna_bundle_check —— LTX 模型包的离厂/到场校验（纯标准库，无需产品运行时）。

为什么单独存在
--------------
``.enc`` 的**读取**只能在冻结（Nuitka）进程内完成：``jasna.model_files.open_tensors``
在非冻结态走 ``safe_open`` 明文分支，而 ``jasna.protection.protected_tensors``
只存在于发布产物内。但「文件对不对、齐不齐、是不是原厂那份」这件事
完全不需要解密——用官方发布的尺寸与 SHA-256 就能判定。

因此本工具做四件事，全程只读、零解密：

1. 按 ``model_files.LTX_DOWNLOADS`` 的官方清单核对每个模型文件的尺寸与摘要；
2. 按 ``bundle_names()`` 的规则判断每个模型变体是否可运行（三个文件齐全）；
3. 检查目录所在文件系统是否满足产品要求（ext4 / Btrfs / XFS）；
4. 输出可归档的验收报告。

命名规则来自 ``jasna/ltx/model_files.py``：明文名 ``<published>``，密文名
``<published>.enc``（``path.with_name(path.name + ".enc")``）。

退出码
------
0  全部通过
1  存在缺失或摘要不符
2  目录里没有任何模型文件（结论不足）
3  用法错误
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TOOL = "jasna-bundle-check"
VERSION = "1.0.0"
REPORT_SCHEMA = "jasna.bundle.report/1"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

#: 官方发布清单：文件名 -> (字节数, sha256)。
#: 直接取自 jasna/ltx/model_files.py 的 LTX_DOWNLOADS；这些都是 **.enc 文件**的值。
PUBLISHED: Dict[str, Tuple[int, str]] = {
    "ltx-restore-alpha1-distill2-int8.safetensors": (
        13_368_048_765,
        "84ef795457d34dae31cac8adeccc9ce0bab68ae8462843aae36ef2df7ee96be1",
    ),
    "ltx-restore-alpha1-distill2-nvfp4.safetensors": (
        7_733_295_013,
        "78ec351e5570984b8c1d851e8d1d2fbc26f6bdefb24cfc6678c8a0ad1c8968bb",
    ),
    "ltx-restore-alpha1-distill4-int8.safetensors": (
        13_368_048_765,
        "d4aaf55e40bcb6af395b8909e5e2232a1573c8f16a78340b5edb1357bb16b158",
    ),
    "ltx-restore-alpha1-distill4-nvfp4.safetensors": (
        7_733_295_013,
        "f7a1a879bb5904649de614c535e0ecd7940c16c014edc6496ed49a5e6211df01",
    ),
    "ltx-restore-alpha1-distill8-int8.safetensors": (
        13_368_048_765,
        "93677e283ec81b0d713581d7a75ffcd35769df8105214b458be3713d9a145c9f",
    ),
    "ltx-restore-alpha1-distill8-nvfp4.safetensors": (
        7_733_295_013,
        "43b8a1cee700dff80a69e2c69caf68a9b47c177ce161fbc8c5f9a0ab42bf82db",
    ),
    "ltx-restore-alpha1-int8.safetensors": (
        13_368_048_765,
        "2892db743471ef1574f4dc47412a1b901a0e02accdc8cbda31c65dbe3c4b06a6",
    ),
    "ltx-restore-alpha1-nvfp4.safetensors": (
        7_733_295_013,
        "71bbbccf96c25f8509fe16da00ef090195e600ad8f467040485bdf75da9aefbd",
    ),
    "vae.safetensors": (
        1_472_223_815,
        "b9252d5a799748d1932d4eed60a2d77461949a19f6feb57f2238dbffe08f5e25",
    ),
    "vae-decoder.safetensors": (
        38_718_609,
        "c60c3572871203bd364d6bb2390225fa4c347f7ea5c64fe468256c900caeea2d",
    ),
}

#: (模型, fast) -> transformer 发布名，取自 _TRANSFORMER_NAMES。
TRANSFORMERS: Dict[Tuple[str, bool], str] = {
    ("distilled2", False): "ltx-restore-alpha1-distill2-int8.safetensors",
    ("distilled2", True): "ltx-restore-alpha1-distill2-nvfp4.safetensors",
    ("distilled4", False): "ltx-restore-alpha1-distill4-int8.safetensors",
    ("distilled4", True): "ltx-restore-alpha1-distill4-nvfp4.safetensors",
    ("distilled", False): "ltx-restore-alpha1-distill8-int8.safetensors",
    ("distilled", True): "ltx-restore-alpha1-distill8-nvfp4.safetensors",
    ("undistilled", False): "ltx-restore-alpha1-int8.safetensors",
    ("undistilled", True): "ltx-restore-alpha1-nvfp4.safetensors",
}

SHARED = ("vae.safetensors", "vae-decoder.safetensors")

SUPPORTED_FS = ("ext4", "btrfs", "xfs")
CONTAINER_FS = ("overlay", "fuse.juicefs", "virtiofs")

#: `.enc` 容器实测结构（在 4090 服务器上用真实文件逐个 chunk 校验过）。
#: 布局：[29 字节头][4 字节块前缀][密文块] × N，头为 magic(9) + nonce(8) + 12 字节保留字段。
ENC_MAGIC = b"JASNAENC1"
ENC_HEADER_LEN = 29
ENC_PREFIX_LEN = 4
MANIFEST_SUFFIX = ".manifest.json"


@dataclass
class ManifestResult:
    """单个 `.enc` 与其 manifest 的一致性结论。"""

    enc: Path
    manifest: Optional[Path] = None
    ok: bool = False
    chunks: int = 0
    checked_chunks: int = 0
    problems: List[str] = field(default_factory=list)
    model_id: str = ""
    signature_bytes: int = 0


@dataclass
class Finding:
    ident: str
    status: str
    detail: str = ""
    remedy: str = ""
    evidence: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.ident,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
            "evidence": self.evidence,
        }


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def mount_fstype(path: Path) -> Optional[str]:
    target = os.path.realpath(str(path))
    best: Optional[Tuple[int, str]] = None
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if "-" not in parts:
                    continue
                index = parts.index("-")
                if index + 1 >= len(parts):
                    continue
                fstype = parts[index + 1]
                mountpoint = parts[4].replace("\\040", " ")
                if target == mountpoint or target.startswith(
                    mountpoint.rstrip("/") + "/"
                ):
                    if best is None or len(mountpoint) > best[0]:
                        best = (len(mountpoint), fstype)
    except OSError:
        return None
    return best[1] if best else None


def resolve(directory: Path, published: str, mode: str) -> Optional[Path]:
    """按 mode 找到某个发布名对应的实际文件。

    auto：优先 ``<name>.enc``（冻结版发布形态），回退明文 ``<name>``；
    enc ：只认 ``<name>.enc``；
    plain：只认 ``<name>``。
    """
    encrypted = directory / (published + ".enc")
    plaintext = directory / published
    if mode == "enc":
        return encrypted if encrypted.is_file() else None
    if mode == "plain":
        return plaintext if plaintext.is_file() else None
    if encrypted.is_file():
        return encrypted
    if plaintext.is_file():
        return plaintext
    return None


def check_environment(findings: List[Finding], directory: Path, allow_container: bool) -> None:
    machine = platform.machine()
    if machine in ("x86_64", "amd64"):
        findings.append(Finding("env.arch", PASS, machine))
    else:
        findings.append(
            Finding(
                "env.arch",
                FAIL,
                machine,
                "发布包只含 x86-64 原生模块；需要对应架构的发行包。",
            )
        )

    fstype = mount_fstype(directory)
    if fstype is None:
        findings.append(
            Finding("env.filesystem", SKIP, "无法解析挂载信息", "确认 /proc 可读。")
        )
    elif fstype in SUPPORTED_FS:
        findings.append(Finding("env.filesystem", PASS, fstype))
    elif allow_container and fstype in CONTAINER_FS:
        findings.append(
            Finding(
                "env.filesystem",
                WARN,
                f"{fstype}（容器/网络文件系统）",
                "宿主卷应为 ext4/Btrfs/XFS；叠加层会影响锁定与落盘行为。",
            )
        )
    else:
        findings.append(
            Finding(
                "env.filesystem",
                FAIL,
                fstype,
                "产品硬性要求 ext4、Btrfs 或 XFS（原文：This Linux build needs an "
                "ext4, Btrfs or XFS disk）。请迁移模型目录。",
            )
        )


def check_files(
    findings: List[Finding],
    directory: Path,
    mode: str,
    names: Sequence[str],
    size_tolerance: int,
    deep: bool,
    manifest_ok: Optional[Dict[str, bool]] = None,
) -> Dict[str, Path]:
    """逐个发布名核对尺寸与摘要；返回 发布名 -> 实际路径。

    ``manifest_ok`` 给出「该文件的 manifest 是否自洽」；仅当摘要不符但 manifest
    自洽时才降级为 WARN，避免把跨版本构建误判成篡改。
    """
    resolved: Dict[str, Path] = {}
    missing: List[str] = []
    size_bad: List[str] = []
    digest_mismatch: List[Tuple[Path, str, str]] = []

    for published in names:
        path = resolve(directory, published, mode)
        if path is None:
            missing.append(published)
            continue
        resolved[published] = path
        expected_size, expected_digest = PUBLISHED[published]
        actual_size = path.stat().st_size
        delta = abs(actual_size - expected_size)
        if delta > size_tolerance:
            size_bad.append(
                f"{path.name}: {actual_size} != {expected_size}"
                f"（差 {actual_size - expected_size:+d} B）"
            )
            continue
        if not deep:
            continue
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            digest_mismatch.append((path, actual_digest, expected_digest))

    degraded = bool(
        digest_mismatch
        and manifest_ok is not None
        and all(manifest_ok.get(path.name) for path, _, _ in digest_mismatch)
    )
    evidence = {
        "resolved": {k: str(v) for k, v in resolved.items()},
        "missing": missing,
        "size_mismatch": size_bad,
        "digest_mismatch": [
            f"{path.name}: {got[:16]}… != {want[:16]}…" for path, got, want in digest_mismatch
        ],
        "deep": deep,
        "digest_mismatch_downgraded": degraded,
    }

    if digest_mismatch and not degraded:
        findings.append(
            Finding(
                "bundle.files",
                FAIL,
                f"{len(digest_mismatch)} 个文件摘要与官方发布不符",
                "文件被改动、下载损坏或版本混用：用官方源重新获取。"
                "若确认是跨版本构建，加 --trust-manifest 让自洽的 manifest 兜底。",
                evidence,
            )
        )
    elif digest_mismatch and degraded:
        findings.append(
            Finding(
                "bundle.files",
                WARN,
                f"{len(digest_mismatch)} 个文件与官方发布摘要不同，但 manifest 自洽",
                "通常表示该文件由另一发行版本产生（v0.11.3 与 v0.12.0 的 .enc 头部"
                "最多差 256 字节）。请与构建该部署的版本核对。",
                evidence,
            )
        )
    elif size_bad:
        findings.append(
            Finding(
                "bundle.files",
                FAIL,
                f"{len(size_bad)} 个文件尺寸不符",
                "尺寸是廉价的前置判据；确认是否跨版本混用"
                "（v0.11.3 与 v0.12.0 的 .enc 头部最多相差 256 字节）。",
                evidence,
            )
        )
    elif missing:
        findings.append(
            Finding(
                "bundle.files",
                FAIL,
                f"缺少 {len(missing)} 个文件",
                "该模型变体不完整；补齐后再验收。",
                evidence,
            )
        )
    else:
        findings.append(
            Finding(
                "bundle.files",
                PASS,
                f"{len(resolved)} 个文件通过"
                + ("尺寸+摘要" if deep else "尺寸")
                + "校验",
                evidence=evidence,
            )
        )

    return resolved


def verify_manifest(enc: Path, verify_chunks: bool) -> ManifestResult:
    """校验 ``<name>.enc`` 与其 ``<name>.enc.manifest.json`` 的一致性。

    结构（在真实文件上逐块验证过）：文件 = 29 字节头 + N × (4 字节前缀 + 密文块)。
    manifest 给出每块的密文长度与 SHA-256，因此**无需解密**即可做块级完整性校验。
    """
    result = ManifestResult(enc=enc)
    manifest_path = enc.with_name(enc.name + MANIFEST_SUFFIX)
    if not manifest_path.is_file():
        result.problems.append("缺少同名 manifest")
        return result
    result.manifest = manifest_path

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = document["payload"]
    except (OSError, ValueError, KeyError) as exc:
        result.problems.append(f"manifest 不可解析: {exc}")
        return result

    result.model_id = str(payload.get("model_id", ""))
    if payload.get("version") != 1:
        result.problems.append(f"manifest version={payload.get('version')!r}，期望 1")
    if not payload.get("chunked"):
        result.problems.append("manifest 未标记 chunked")

    try:
        signature = base64.b64decode(document.get("signature", ""))
        result.signature_bytes = len(signature)
    except Exception:  # noqa: BLE001
        result.signature_bytes = 0
    if result.signature_bytes != 64:
        result.problems.append(f"签名字节数 {result.signature_bytes}，期望 64")

    try:
        header = base64.b64decode(payload["header"])
    except Exception as exc:  # noqa: BLE001
        result.problems.append(f"header 解码失败: {exc}")
        return result
    if len(header) != ENC_HEADER_LEN:
        result.problems.append(f"header 长度 {len(header)}，期望 {ENC_HEADER_LEN}")
        return result
    if header[:9] != ENC_MAGIC:
        result.problems.append(f"header magic {header[:9]!r}，期望 {ENC_MAGIC!r}")
        return result

    lengths = list(payload.get("lengths") or [])
    hashes = list(payload.get("hashes") or [])
    result.chunks = len(lengths)
    if len(hashes) != len(lengths):
        result.problems.append(f"lengths={len(lengths)} 与 hashes={len(hashes)} 数量不一致")
        return result

    size = enc.stat().st_size
    expected = ENC_HEADER_LEN + ENC_PREFIX_LEN * len(lengths) + sum(lengths)
    if size != expected:
        result.problems.append(f"文件尺寸 {size} != 头部推算 {expected}")
    if payload.get("size") not in (None, size):
        result.problems.append(f"manifest.size={payload.get('size')} != 实际 {size}")

    with enc.open("rb") as handle:
        if handle.read(ENC_HEADER_LEN) != header:
            result.problems.append("磁盘头与 manifest.header 不一致")
        for index, length in enumerate(lengths):
            if len(handle.read(ENC_PREFIX_LEN)) != ENC_PREFIX_LEN:
                result.problems.append(f"块 {index}: 前缀短读")
                break
            if not verify_chunks:
                handle.seek(length, os.SEEK_CUR)
                continue
            body = handle.read(length)
            if len(body) != length:
                result.problems.append(f"块 {index}: 短读 {len(body)}/{length}")
                break
            if hashlib.sha256(body).hexdigest() != hashes[index]:
                result.problems.append(f"块 {index}: SHA-256 与 manifest 不符")
                break
            result.checked_chunks += 1
        if handle.read(1):
            result.problems.append("文件尾部存在多余字节")

    result.ok = not result.problems
    return result


def check_manifests(
    findings: List[Finding], directory: Path, verify_chunks: bool
) -> Dict[str, bool]:
    """对目录内所有 ``.enc`` 做 manifest 校验，返回 文件名 -> 是否自洽。"""
    enc_files = sorted(directory.glob("*.enc"))
    if not enc_files:
        findings.append(Finding("bundle.manifest", SKIP, "目录内没有 .enc"))
        return {}

    results: List[ManifestResult] = []
    mapping: Dict[str, bool] = {}
    for enc in enc_files:
        try:
            item = verify_manifest(enc, verify_chunks)
        except Exception as exc:  # noqa: BLE001
            item = ManifestResult(enc=enc, problems=[f"校验异常: {type(exc).__name__}: {exc}"])
        results.append(item)
        mapping[enc.name] = item.ok

    bad = [item for item in results if not item.ok]
    evidence = {
        "verify_chunks": verify_chunks,
        "files": [
            {
                "file": item.enc.name,
                "ok": item.ok,
                "chunks": item.chunks,
                "checked_chunks": item.checked_chunks,
                "model_id": item.model_id,
                "signature_bytes": item.signature_bytes,
                "problems": item.problems,
            }
            for item in results
        ],
    }

    if bad:
        findings.append(
            Finding(
                "bundle.manifest",
                FAIL,
                f"{len(bad)}/{len(results)} 个 .enc 与 manifest 不自洽",
                "块级摘要或容器布局不符：文件被截断/损坏/替换。manifest 是本地权威元数据，"
                "签名验证需在冻结进程内用产品公钥完成。",
                evidence,
            )
        )
    else:
        total_chunks = sum(item.checked_chunks for item in results)
        findings.append(
            Finding(
                "bundle.manifest",
                PASS,
                f"{len(results)} 个 .enc 与 manifest 自洽"
                + (f"（已校验 {total_chunks} 个块的密文摘要）" if verify_chunks else "（仅结构校验）"),
                remedy="" if verify_chunks else "加 --verify-chunks 可做块级密文摘要校验。",
                evidence=evidence,
            )
        )
    return mapping


def check_variants(
    findings: List[Finding], directory: Path, mode: str, present: int
) -> None:
    """按 bundle_names() 规则判断每个模型变体是否可运行。"""
    runnable: List[str] = []
    incomplete: Dict[str, List[str]] = {}

    for (model, fast), transformer in TRANSFORMERS.items():
        label = f"{model}/{'nvfp4' if fast else 'int8'}"
        needed = (transformer, *SHARED)
        absent = [name for name in needed if resolve(directory, name, mode) is None]
        if absent:
            incomplete[label] = absent
        else:
            runnable.append(label)

    evidence = {"runnable": runnable, "incomplete": incomplete}
    if present == 0:
        findings.append(
            Finding(
                "bundle.variants",
                SKIP,
                "目录内没有任何已发布的模型文件",
                "先部署模型文件（发布名或 <发布名>.enc），或确认 --root 指向正确目录。",
                evidence,
            )
        )
        return
    if runnable:
        findings.append(
            Finding(
                "bundle.variants",
                PASS,
                f"{len(runnable)}/{len(TRANSFORMERS)} 个变体文件齐全：{', '.join(runnable)}",
                evidence=evidence,
            )
        )
    elif incomplete:
        findings.append(
            Finding(
                "bundle.variants",
                FAIL,
                "没有任何模型变体文件齐全",
                "每个变体需要 1 个 transformer + vae + vae-decoder 共 3 个文件。",
                evidence,
            )
        )
    else:
        findings.append(Finding("bundle.variants", SKIP, "未发现任何模型文件", evidence=evidence))


def check_unknown(findings: List[Finding], directory: Path) -> None:
    """列出不在官方清单内的 .enc 资产。

    ``unet-4x.onnx.enc`` 这类附加模型不在 ``LTX_DOWNLOADS`` 里，无法用官方摘要验收；
    静默忽略会让验收报告看起来「全过」，所以必须显式列出。
    """
    unknown: List[Dict[str, object]] = []
    for path in sorted(directory.glob("*.enc")):
        base = path.name[: -len(".enc")]
        if base in PUBLISHED:
            continue
        unknown.append({"name": path.name, "bytes": path.stat().st_size})

    if unknown:
        findings.append(
            Finding(
                "bundle.extra",
                WARN,
                f"{len(unknown)} 个 .enc 不在官方清单内",
                "这些是附加资产（例如 unet-4x.onnx.enc），无法用官方摘要验收；"
                "请在验收单中单列，并核对产品版本是否应包含它们。",
                {"unknown": unknown},
            )
        )
    else:
        findings.append(Finding("bundle.extra", PASS, "无清单外的 .enc"))


def check_mixed_layout(findings: List[Finding], directory: Path) -> None:
    """同一个发布名同时存在 .enc 与明文时给出提示。

    ``open_tensors`` 在冻结态一律走保护路径，明文会被忽略；在源码态则相反。
    混放会让不同运行模式读到不同文件，是验收事故的常见来源。
    """
    both: List[str] = []
    for published in PUBLISHED:
        if (directory / published).is_file() and (
            directory / (published + ".enc")
        ).is_file():
            both.append(published)
    if both:
        findings.append(
            Finding(
                "bundle.layout",
                WARN,
                f"{len(both)} 个发布名同时存在 .enc 与明文",
                "冻结发行版只读 .enc，源码态只读明文；混放会导致不同模式读到不同文件。"
                "建议每台机器只保留一种形态。",
                {"pairs": both},
            )
        )
    else:
        findings.append(Finding("bundle.layout", PASS, "无明文/密文混放"))


def render(findings: List[Finding], verbose: bool) -> str:
    lines = [f"{TOOL} {VERSION}", "=" * 78]
    for item in findings:
        lines.append(f"[{item.status:4}] {item.ident}")
        if item.detail:
            lines.append(f"       {item.detail}")
        if item.remedy and (verbose or item.status in (FAIL, SKIP)):
            lines.append(f"       -> {item.remedy}")
        if verbose and item.evidence:
            lines.append(f"       evidence: {json.dumps(item.evidence, ensure_ascii=False)}")
    counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for item in findings:
        counts[item.status] = counts.get(item.status, 0) + 1
    lines.append("-" * 78)
    lines.append(
        "汇总: " + ", ".join(f"{k}={counts.get(k, 0)}" for k in (PASS, WARN, FAIL, SKIP))
    )
    if counts[FAIL]:
        code = 1
    elif counts[SKIP]:
        code = 2
    else:
        code = 0
    verdict = {0: "通过", 1: "不通过", 2: "结论不足"}[code]
    lines.append(f"结论: {verdict}（退出码 {code}）")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="按官方发布清单校验 LTX 模型包（只读、不解密）。",
    )
    parser.add_argument("--version", action="version", version=f"{TOOL} {VERSION}")
    parser.add_argument("--root", default=".", help="模型文件所在目录（默认当前目录）")
    parser.add_argument(
        "--mode",
        choices=("auto", "enc", "plain"),
        default="auto",
        help="auto 优先 .enc 再回退明文；enc/plain 只认单一种形态",
    )
    parser.add_argument(
        "--size-tolerance",
        type=int,
        default=0,
        help="允许的尺寸偏差字节数；跨 v0.11.3/v0.12.0 验收可设 256",
    )
    parser.add_argument(
        "--no-deep",
        dest="deep",
        action="store_false",
        help="跳过 SHA-256，只做尺寸检查（13 GB 全量摘要约需数分钟）",
    )
    parser.add_argument(
        "--verify-chunks",
        action="store_true",
        help="按 manifest 逐块校验密文 SHA-256（最强的不解密完整性判定）",
    )
    parser.add_argument(
        "--trust-manifest",
        action="store_true",
        help="官方发布摘要不符但 manifest 自洽时降级为 WARN（跨版本构建场景）",
    )
    parser.add_argument(
        "--allow-container", action="store_true", help="允许 overlay 等容器文件系统"
    )
    parser.add_argument("--report", help="JSON 报告输出路径")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    directory = Path(args.root).resolve()
    if not directory.is_dir():
        print(f"错误: {directory} 不是目录", file=sys.stderr)
        return 3

    findings: List[Finding] = []
    check_environment(findings, directory, args.allow_container)
    manifest_ok = check_manifests(findings, directory, args.verify_chunks)
    resolved = check_files(
        findings,
        directory,
        args.mode,
        list(PUBLISHED),
        args.size_tolerance,
        args.deep,
        manifest_ok if args.trust_manifest else None,
    )
    check_variants(findings, directory, args.mode, len(resolved))
    check_unknown(findings, directory)
    check_mixed_layout(findings, directory)

    text = render(findings, args.verbose)
    print(text)

    if args.report:
        counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for item in findings:
            counts[item.status] = counts.get(item.status, 0) + 1
        exit_code = 1 if counts[FAIL] else (2 if counts[SKIP] else 0)
        payload = {
            "schema": REPORT_SCHEMA,
            "tool": TOOL,
            "version": VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target": {
                "root": str(directory),
                "mode": args.mode,
                "deep": args.deep,
                "size_tolerance": args.size_tolerance,
            },
            "summary": counts,
            "exit_code": exit_code,
            "findings": [item.as_dict() for item in findings],
        }
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n报告已写入: {out}")

    counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for item in findings:
        counts[item.status] = counts.get(item.status, 0) + 1
    return 1 if counts[FAIL] else (2 if counts[SKIP] else 0)


if __name__ == "__main__":
    sys.exit(main())
