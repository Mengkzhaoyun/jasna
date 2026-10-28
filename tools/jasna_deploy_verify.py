#!/usr/bin/env python3
"""jasna-deploy-verify —— Jasna 离线部署与现场验收校验器。

设计原则
--------
1. 只读。除 ``record`` 写基线、``--report`` 写报告外，不修改部署目录。
2. 不实现任何密码学。许可证与完整性判定一律委托产品自身的 ``_core``；
   本工具不写入、不导出任何明文权重。
3. 分层判定：与产品无关的硬性条件（ABI、架构、文件系统、清单摘要、
   权限、ELF 有效性）作为验收门槛；依赖产品 API 的判定作为建议项，
   拿不到就报 SKIP，绝不猜。

退出码
------
0  通过（允许 WARN）
1  不通过（存在 FAIL）
2  结论不足（存在必需的 SKIP）
3  用法或内部错误

典型用法
--------
出厂/交付前采集基线::

    <jasna>/python/bin/python3 jasna_deploy_verify.py record \\
        --root /opt/jasna --out baseline.json

客户现场验收::

    <jasna>/python/bin/python3 jasna_deploy_verify.py verify \\
        --root /opt/jasna --license /etc/jasna/license.bin \\
        --baseline baseline.json --report /tmp/acceptance.json --strict
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import platform
import signal
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOOL = "jasna-deploy-verify"
VERSION = "1.0.0"
BASELINE_SCHEMA = "jasna.deploy.baseline/1"
REPORT_SCHEMA = "jasna.deploy.report/1"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

#: `_core` 支持的 ABI。扩展模块名为 cpython-313，故解释器必须是 3.13.x。
EXPECTED_PY = (3, 13)

#: `_core` 中硬编码的文件系统要求（字符串 @0x20d336f）。
SUPPORTED_FS = ("ext4", "btrfs", "xfs")
CONTAINER_FS = ("overlay", "fuse.juicefs", "virtiofs")

#: `_core` 用来判断“是否处于受支持的用户命名空间”的字面量
#: （memcmp 长度 0x21，@0x1ff... 反汇编可见）。
EXPECTED_UID_MAP = "         0          0 4294967295\n"

#: `_core` 抛出的 ImportError 文案 → 现场可执行的处置建议。
#: 全部取自二进制字符串（observed）。
IMPORT_ERROR_HINTS: Tuple[Tuple[str, str], ...] = (
    (
        "could not capture the worker process environment",
        "环境采集失败：清理登录会话与预加载库后重试（干净终端/桌面会话）。",
    ),
    (
        "unsafe Jasna launch environment",
        "启动环境不可信：检查 LD_PRELOAD、注入式反病毒/覆盖层、以及父进程链。",
    ),
    (
        "the Jasna runtime failed its integrity check",
        "运行时完整性校验失败：重新释放完整发行包，勿混用不同版本文件。",
    ),
    (
        "installed Jasna files failed verification",
        "安装文件清单校验失败：文件缺失或被改动，需重新释放发行包。",
    ),
    (
        "unsigned or modified Jasna runtime",
        "运行时未签名或被修改：确认使用正式签名的发行包。",
    ),
    (
        "worker admission failed",
        "worker 准入失败：检查 seccomp/ptrace 限制与容器安全配置。",
    ),
    (
        "unauthenticated LTX worker launch",
        "LTX worker 未通过认证：确认可执行文件为部署包内的原件。",
    ),
    (
        "could not open the Jasna folder",
        "无法定位 Jasna 目录：请用产品自带解释器启动本工具，不要用系统 python3。",
    ),
    (
        "needs an ext4, Btrfs or XFS disk",
        "所处磁盘文件系统不受支持：把 Jasna 目录迁移到 ext4/Btrfs/XFS 卷。",
    ),
)

#: 采集基线时默认忽略的易变路径（可用 --exclude 覆盖）。
DEFAULT_EXCLUDES = (
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.log",
    ".cache/*",
    "logs/*",
    "tmp/*",
    "*.tmp",
)


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """单项校验结果。"""

    ident: str
    status: str
    detail: str = ""
    remedy: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.ident,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
            "evidence": self.evidence,
        }


class Report:
    """校验结果集合与汇总。"""

    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(
        self,
        ident: str,
        status: str,
        detail: str = "",
        remedy: str = "",
        **evidence: Any,
    ) -> Check:
        item = Check(ident, status, detail, remedy, evidence)
        self.checks.append(item)
        return item

    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def exit_code(self) -> int:
        counts = self.counts()
        if counts.get(FAIL):
            return 1
        if counts.get(SKIP):
            return 2
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "tool": TOOL,
            "version": VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": self.counts(),
            "exit_code": self.exit_code(),
            "checks": [c.as_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# 环境探测
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_any(rel: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def iter_files(root: Path, excludes: Iterable[str]) -> Iterable[Path]:
    """按稳定顺序产出 root 下的普通文件，跳过符号链接与排除项。"""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            if full.is_symlink() or not full.is_file():
                continue
            rel = full.relative_to(root).as_posix()
            if matches_any(rel, excludes):
                continue
            yield full


def mount_fstype(path: Path) -> Optional[str]:
    """返回包含 path 的挂载点文件系统类型（读 /proc/self/mountinfo）。"""
    target = os.path.realpath(str(path))
    best: Optional[Tuple[int, str]] = None
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if "-" not in parts:
                    continue
                idx = parts.index("-")
                if idx + 1 >= len(parts):
                    continue
                fstype = parts[idx + 1]
                mountpoint = parts[4].replace("\\040", " ")
                if target == mountpoint or target.startswith(
                    mountpoint.rstrip("/") + "/"
                ):
                    if best is None or len(mountpoint) > best[0]:
                        best = (len(mountpoint), fstype)
    except OSError:
        return None
    return best[1] if best else None


def is_elf_x86_64(path: Path) -> Tuple[bool, str]:
    """轻量 ELF 头校验：64 位、小端、x86-64、ET_DYN/ET_EXEC。"""
    try:
        with path.open("rb") as fh:
            head = fh.read(20)
    except OSError as exc:
        return False, f"读取失败: {exc}"
    if len(head) < 20:
        return False, "文件过短，不是 ELF"
    if head[:4] != b"\x7fELF":
        return False, "缺少 ELF magic"
    ei_class, ei_data = head[4], head[5]
    if ei_class != 2:
        return False, f"非 64 位 ELF (EI_CLASS={ei_class})"
    if ei_data != 1:
        return False, f"非小端 ELF (EI_DATA={ei_data})"
    e_type = int.from_bytes(head[16:18], "little")
    e_machine = int.from_bytes(head[18:20], "little")
    if e_machine != 0x3E:
        return False, f"非 x86-64 (e_machine=0x{e_machine:x})"
    if e_type not in (2, 3):
        return False, f"非可执行/共享对象 (e_type={e_type})"
    return True, "ELF64 x86-64"


def read_uid_map() -> Optional[str]:
    try:
        with open("/proc/self/uid_map", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def process_exe() -> Optional[str]:
    try:
        return os.path.realpath("/proc/self/exe")
    except OSError:
        return None


@contextlib.contextmanager
def time_limit(seconds: int, what: str) -> Iterable[None]:
    """用 SIGALRM 给阻塞的原生调用加硬超时（仅主线程可用）。"""

    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(signum, frame):  # noqa: ANN001
        raise TimeoutError(f"{what} 超过 {seconds}s 未返回")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# 产品 API 适配层
# ---------------------------------------------------------------------------


class CoreAPI:
    """对 `jasna.protection._core` 的惰性、失败安全封装。

    只调用只读/校验类接口；不触碰 decrypt_* 与任何权重导出接口。
    """

    #: 允许调用的接口白名单（其余一律不碰）。
    ALLOWED = frozenset(
        {
            "machine_key",
            "authorize",
            "matches_machine",
            "cache_tag",
            "cache_verify",
            "extension_verified",
            "create_code_cache",
            "check_ltx_worker_mappings",
            "seal_ltx_worker_mappings",
            "launch_ltx_worker",
        }
    )

    def __init__(self) -> None:
        self.module: Any = None
        self.import_error: Optional[BaseException] = None

    def load(self) -> bool:
        if self.module is not None:
            return True
        try:
            from jasna.protection import _core  # type: ignore import-not-found

            self.module = _core
            return True
        except BaseException as exc:  # noqa: BLE001 - 导入期可能抛 SystemExit
            self.import_error = exc
            return False

    def has(self, name: str) -> bool:
        return (
            name in self.ALLOWED
            and self.module is not None
            and callable(getattr(self.module, name, None))
        )

    def call(self, name: str, *args: Any) -> Tuple[bool, Any]:
        """返回 (是否成功, 结果或异常)。"""
        if not self.has(name):
            return False, RuntimeError(f"接口不可用: {name}")
        try:
            return True, getattr(self.module, name)(*args)
        except BaseException as exc:  # noqa: BLE001
            return False, exc

    def explain_import_error(self) -> Tuple[str, str]:
        """把导入失败映射为 (详情, 处置建议)。"""
        exc = self.import_error
        if exc is None:
            return "", ""
        text = str(exc)
        for needle, hint in IMPORT_ERROR_HINTS:
            if needle in text:
                return text, hint
        if isinstance(exc, ModuleNotFoundError):
            return (
                f"{type(exc).__name__}: {text}",
                "产品包不在解释器搜索路径内：用产品自带解释器并把部署根目录加入 "
                "PYTHONPATH（jasna 包应位于 <root>/…/jasna/protection/ 下）。",
            )
        return f"{type(exc).__name__}: {text}", "对照产品日志确认具体闸门；必要时收集完整回溯。"


# ---------------------------------------------------------------------------
# 校验项
# ---------------------------------------------------------------------------


def check_python(report: Report) -> None:
    ver = sys.version_info
    detail = f"Python {ver.major}.{ver.minor}.{ver.micro} ({platform.machine()})"
    if (ver.major, ver.minor) == EXPECTED_PY:
        report.add("env.python_abi", PASS, detail, abi="cpython-%d%d" % EXPECTED_PY)
    else:
        report.add(
            "env.python_abi",
            FAIL,
            detail,
            "扩展模块编译于 CPython 3.13。请用产品自带解释器运行本工具，"
            "不要用系统 python3。",
            expected="3.13",
            actual=f"{ver.major}.{ver.minor}",
        )


def check_arch(report: Report) -> None:
    machine = platform.machine()
    if machine in ("x86_64", "amd64"):
        report.add("env.arch", PASS, machine)
    else:
        report.add(
            "env.arch",
            FAIL,
            machine,
            "发行包只含 x86-64 原生模块；请改用 x86-64 主机或对应架构的发行包。",
            expected="x86_64",
        )


def check_filesystem(report: Report, root: Path, allow_container: bool) -> None:
    fstype = mount_fstype(root)
    if fstype is None:
        report.add(
            "env.filesystem",
            SKIP,
            f"无法解析 {root} 的挂载信息",
            "确认 /proc 已挂载且可读。",
        )
        return
    if fstype in SUPPORTED_FS:
        report.add("env.filesystem", PASS, fstype, mount_root=str(root))
        return
    if allow_container and fstype in CONTAINER_FS:
        report.add(
            "env.filesystem",
            WARN,
            f"{fstype}（容器/网络文件系统）",
            "宿主卷应为 ext4/Btrfs/XFS；容器叠加层可能影响锁定与落盘行为。",
        )
        return
    report.add(
        "env.filesystem",
        FAIL,
        fstype,
        "产品硬性要求 ext4、Btrfs 或 XFS。请把 Jasna 目录迁移到受支持卷"
        "（官方文案：This Linux build needs an ext4, Btrfs or XFS disk）。",
        observed=fstype,
        supported=list(SUPPORTED_FS),
    )


def check_uid_map(report: Report) -> None:
    content = read_uid_map()
    if content is None:
        report.add("env.userns", SKIP, "/proc/self/uid_map 不可读")
        return
    if content == EXPECTED_UID_MAP:
        report.add("env.userns", PASS, "单一映射，覆盖完整 UID 范围")
    else:
        report.add(
            "env.userns",
            WARN,
            content.strip().replace("\n", " | "),
            "与 _core 判定的字面量不一致。若运行正常可忽略；"
            "若许可证判定失败，优先排查用户命名空间映射。",
            expected=EXPECTED_UID_MAP.strip(),
        )


def check_bundle(report: Report, root: Path, core: CoreAPI) -> None:
    """确认本工具确实跑在产品自带的解释器/目录里。"""
    exe = process_exe()
    detail = {
        "python": sys.executable,
        "proc_exe": exe,
        "root": str(root),
    }
    inside = bool(exe) and os.path.realpath(exe).startswith(
        os.path.realpath(str(root)) + os.sep
    )
    if inside:
        report.add("env.bundle", PASS, f"/proc/self/exe 位于部署根目录内", **detail)
        return
    remedy = (
        "必须在冻结发行版进程内运行：发行版是 Nuitka 构建（is_frozen() 检查 "
        "sys.frozen / __compiled__），没有可借用的独立解释器。请把本工具作为隐藏子命令"
        "编入发行版（例如 `jasna verify-deployment`）再由 jasna 可执行文件启动；"
        "否则 _core 会把 dirname(/proc/self/exe) 当作 Jasna 目录而拒绝加载。"
    )
    report.add("env.bundle", WARN, "本进程不在部署根目录内", remedy, **detail)


def load_baseline(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("schema") != BASELINE_SCHEMA:
        return None
    return data


def build_baseline(root: Path, excludes: Iterable[str]) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for full in iter_files(root, excludes):
        st = full.stat()
        files[full.relative_to(root).as_posix()] = {
            "sha256": sha256_file(full),
            "size": st.st_size,
            "mode": st.st_mode & 0o777,
        }
    return {
        "schema": BASELINE_SCHEMA,
        "tool": TOOL,
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "excludes": list(excludes),
        "files": files,
    }


def check_digests(
    report: Report,
    root: Path,
    baseline: Dict[str, Any],
    strict: bool,
) -> None:
    expected: Dict[str, Any] = baseline.get("files", {})
    actual_paths = list(iter_files(root, baseline.get("excludes", DEFAULT_EXCLUDES)))
    actual_rel = {p.relative_to(root).as_posix() for p in actual_paths}

    missing = sorted(set(expected) - actual_rel)
    added = sorted(actual_rel - set(expected))

    changed: List[str] = []
    mode_changed: List[str] = []
    for rel in sorted(set(expected) & actual_rel):
        full = root / rel
        want = expected[rel]
        try:
            digest = sha256_file(full)
            mode = full.stat().st_mode & 0o777
        except OSError as exc:
            changed.append(f"{rel} (读取失败: {exc})")
            continue
        if digest != want.get("sha256"):
            changed.append(rel)
        elif mode != want.get("mode"):
            mode_changed.append(f"{rel} ({want.get('mode'):o} -> {mode:o})")

    evidence = {
        "expected_files": len(expected),
        "actual_files": len(actual_rel),
        "missing": missing,
        "added": added,
        "changed": changed,
        "mode_changed": mode_changed,
    }

    if missing:
        report.add(
            "deploy.integrity",
            FAIL,
            f"缺失 {len(missing)} 个文件",
            "部署不完整：重新释放发行包，勿手工裁剪。",
            **evidence,
        )
    elif changed:
        report.add(
            "deploy.integrity",
            FAIL,
            f"{len(changed)} 个文件摘要与基线不符",
            "文件被改动或版本混用：以基线对应的发行包重新部署。",
            **evidence,
        )
    elif added:
        report.add(
            "deploy.integrity",
            FAIL if strict else WARN,
            f"{len(added)} 个基线外文件",
            "严格模式下视为污染；如为运行产物请加入 --exclude。",
            **evidence,
        )
    elif mode_changed:
        report.add(
            "deploy.integrity",
            WARN,
            f"{len(mode_changed)} 个文件权限位变化",
            "原生模块需保留可执行/读权限；确认拷贝过程未丢权限。",
            **evidence,
        )
    else:
        report.add(
            "deploy.integrity",
            PASS,
            f"{len(actual_rel)} 个文件与基线一致",
            **evidence,
        )


def check_permissions(report: Report, root: Path) -> None:
    world_writable: List[str] = []
    setuid: List[str] = []
    unreadable: List[str] = []
    for full in iter_files(root, DEFAULT_EXCLUDES):
        rel = full.relative_to(root).as_posix()
        try:
            mode = full.stat().st_mode
        except OSError:
            continue
        if mode & stat.S_IWOTH:
            world_writable.append(rel)
        if mode & (stat.S_ISUID | stat.S_ISGID):
            setuid.append(rel)
        if not os.access(full, os.R_OK):
            unreadable.append(rel)

    evidence = {
        "world_writable": world_writable[:32],
        "setuid": setuid[:32],
        "unreadable": unreadable[:32],
    }
    if unreadable:
        report.add(
            "deploy.permissions",
            FAIL,
            f"{len(unreadable)} 个文件当前用户不可读",
            "以部署所有者身份运行验收，或修正属主/权限。",
            **evidence,
        )
    elif setuid:
        report.add(
            "deploy.permissions",
            FAIL,
            f"{len(setuid)} 个文件带 setuid/setgid 位",
            "发行包不应包含 setuid 文件；确认是否被篡改。",
            **evidence,
        )
    elif world_writable:
        report.add(
            "deploy.permissions",
            WARN,
            f"{len(world_writable)} 个文件全局可写",
            "全局可写会削弱完整性保证，建议 chmod o-w。",
            **evidence,
        )
    else:
        report.add("deploy.permissions", PASS, "无 setuid / 全局可写 / 不可读文件")


def check_native_modules(report: Report, root: Path, core: CoreAPI) -> None:
    modules = [p for p in iter_files(root, DEFAULT_EXCLUDES) if p.suffix == ".so"]
    if not modules:
        report.add(
            "native.modules",
            FAIL,
            "部署目录内没有任何 .so",
            "发行包至少应包含 _core.so 与 _opaque_ltx*.so。",
        )
        return

    bad: List[str] = []
    for full in modules:
        ok, why = is_elf_x86_64(full)
        if not ok:
            bad.append(f"{full.relative_to(root).as_posix()}: {why}")

    names = sorted(p.relative_to(root).as_posix() for p in modules)
    evidence: Dict[str, Any] = {"count": len(modules), "invalid": bad, "modules": names[:64]}

    if bad:
        report.add(
            "native.modules",
            FAIL,
            f"{len(bad)} 个原生模块不是有效 ELF x86-64",
            "文件损坏或被替换；重新部署。",
            **evidence,
        )
        return

    core_present = any("_core.so" in n for n in names)
    ltx_present = any("_opaque_ltx" in n and n.endswith(".so") for n in names)
    if not (core_present and ltx_present):
        report.add(
            "native.modules",
            FAIL,
            "缺少核心模块（_core.so 或 _opaque_ltx*.so）",
            "部署不完整。",
            **evidence,
        )
        return

    verified: Optional[bool] = None
    if core.has("extension_verified"):
        results = {}
        for full in modules:
            ok, value = core.call("extension_verified", str(full))
            results[full.relative_to(root).as_posix()] = (
                bool(value) if ok else f"error: {value}"
            )
        evidence["extension_verified"] = results
        rejected = [k for k, v in results.items() if v is False]
        if rejected:
            report.add(
                "native.modules",
                FAIL,
                f"产品拒绝 {len(rejected)} 个原生模块",
                "文件不在签名清单内：使用正式发行包。",
                **evidence,
            )
            return
        verified = True

    report.add(
        "native.modules",
        PASS,
        f"{len(modules)} 个 ELF 模块有效"
        + ("，且通过产品 verification" if verified else "（产品接口未调用）"),
        **evidence,
    )


def check_license(
    report: Report,
    core: CoreAPI,
    license_path: Optional[Path],
    machine_key_override: Optional[str],
    cloud: bool,
) -> None:
    if license_path is None:
        report.add(
            "license.authorization",
            SKIP,
            "未提供 --license",
            "验收必须提供许可证文件，否则无法判定授权有效性。",
        )
        return

    # 必须先加载：模块导入本身就是第一道闸门，且后续 check_runtime 会复用结果。
    core.load()

    if not core.has("machine_key") or not core.has("authorize"):
        detail, remedy = core.explain_import_error()
        report.add(
            "license.authorization",
            SKIP,
            "产品接口不可用" + (f"：{detail}" if detail else ""),
            remedy or "先修复 env.* 项再执行授权校验。",
        )
        return

    try:
        license_bytes = license_path.read_bytes()
    except OSError as exc:
        report.add(
            "license.authorization",
            FAIL,
            f"许可证不可读: {exc}",
            "确认路径与权限（通常为 /etc/jasna/license.bin）。",
        )
        return

    ok, key = core.call("machine_key", bool(cloud))
    if not ok:
        report.add(
            "license.machine_key",
            FAIL,
            f"machine_key 失败: {key}",
            "按异常文案比对 IMPORT_ERROR_HINTS 中的处置建议。",
        )
        return

    key_str = machine_key_override if machine_key_override is not None else str(key)
    report.add(
        "license.machine_key",
        PASS,
        f"{len(key_str)} 字符" + ("（命令行覆盖）" if machine_key_override else ""),
        machine_key_sha256=hashlib.sha256(key_str.encode("utf-8")).hexdigest(),
        length=len(key_str),
    )

    key_bytes = key_str.encode("utf-8", "surrogateescape")
    if len(key_bytes) != 68:
        report.add(
            "license.binding_probe",
            SKIP,
            f"机器码为 {len(key_bytes)} 字节，authorize 要求 68",
            "你的发行版可能用了不同的编码（如 hex/base64）。"
            "请用 --machine-key 指定实际传给 authorize 的形态。",
            expected_bytes=68,
            actual_bytes=len(key_bytes),
        )
    else:
        ok, verdict = core.call("authorize", key_bytes, license_bytes)
        if ok and bool(verdict):
            report.add(
                "license.authorization",
                PASS,
                "许可证在本机校验通过",
                license_sha256=hashlib.sha256(license_bytes).hexdigest(),
                bytes=len(license_bytes),
            )
        elif ok:
            report.add(
                "license.authorization",
                FAIL,
                "许可证被拒绝",
                "三种常见原因：绑定到其他机器、已过期、签发对象不匹配。",
                license_sha256=hashlib.sha256(license_bytes).hexdigest(),
            )
        else:
            report.add(
                "license.authorization",
                SKIP,
                f"authorize 抛出异常: {verdict}",
                "接口契约可能与本文假设不同（是否需要先 open_session）；"
                "请把异常原文反馈以校准。",
            )


def check_runtime(
    report: Report, core: CoreAPI, probe_worker: bool, timeout_s: int
) -> None:
    if not core.load():
        detail, remedy = core.explain_import_error()
        report.add(
            "runtime.import_core",
            FAIL,
            detail or "无法导入 jasna.protection._core",
            remedy or "确认用产品自带解释器、部署完整、磁盘为受支持文件系统。",
        )
        return

    report.add("runtime.import_core", PASS, "_core 导入成功，闸门链已放行")

    try:
        from jasna._frozen import is_frozen as _is_frozen

        frozen: Optional[bool] = bool(_is_frozen())
    except Exception:  # noqa: BLE001
        frozen = None
    if frozen is True:
        report.add("runtime.frozen", PASS, "Nuitka 冻结进程，.enc 保护路径可用")
    elif frozen is False:
        report.add(
            "runtime.frozen",
            FAIL,
            "非冻结进程",
            ".enc 只能由冻结产物读取；源码态下 open_tensors 走明文 safe_open 分支。"
            "请把本工具编入发行版后运行。",
        )
    else:
        report.add("runtime.frozen", SKIP, "无法判定 is_frozen()", "检查 jasna._frozen 是否可导入。")

    if core.has("check_ltx_worker_mappings"):
        ok, value = core.call("check_ltx_worker_mappings")
        report.add(
            "runtime.mappings",
            PASS if ok else FAIL,
            "映射策略通过" if ok else f"失败: {value}",
            "检查是否有预加载/注入工具改了可执行映射。",
        )

    if core.has("create_code_cache"):
        ok, value = core.call("create_code_cache")
        if ok:
            path = str(value)
            writable = os.path.isdir(path) and os.access(path, os.W_OK)
            report.add(
                "runtime.code_cache",
                PASS if writable else WARN,
                f"{path}（{'可写' if writable else '不可写'}）",
                "" if writable else "首次启动需要可写缓存目录以保存签名 PTX。",
            )
        else:
            report.add(
                "runtime.code_cache",
                WARN,
                f"create_code_cache 失败: {value}",
                "非致命；但会失去 GPU 编译缓存复用。",
            )

    if not probe_worker:
        report.add(
            "runtime.worker",
            SKIP,
            "未启用 --runtime-probe",
            "验收若需覆盖 LTX worker，请加 --runtime-probe（会短暂启动子进程）。",
        )
        return

    if not core.has("launch_ltx_worker"):
        report.add("runtime.worker", SKIP, "launch_ltx_worker 不可用")
        return

    try:
        with time_limit(timeout_s, "launch_ltx_worker"):
            ok, value = core.call("launch_ltx_worker")
    except TimeoutError as exc:
        report.add(
            "runtime.worker",
            FAIL,
            str(exc),
            "worker 未在期限内完成准入握手：检查 seccomp、pidfd、"
            "管道权限，或适度提高 --worker-timeout。",
            timeout_s=timeout_s,
        )
        return

    if ok:
        report.add(
            "runtime.worker",
            PASS,
            "worker 启动并完成准入握手",
            result=str(value)[:200],
        )
    else:
        report.add(
            "runtime.worker",
            FAIL,
            f"worker 启动失败: {value}",
            "检查 seccomp、pidfd 可用性、/proc 可读性与超时限制。",
        )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def render_text(report: Report, verbose: bool) -> str:
    lines: List[str] = []
    lines.append(f"{TOOL} {VERSION}")
    lines.append("=" * 78)
    for check in report.checks:
        lines.append(f"[{check.status:4}] {check.ident}")
        if check.detail:
            lines.append(f"       {check.detail}")
        if check.remedy and (verbose or check.status in (FAIL, SKIP)):
            lines.append(f"       -> {check.remedy}")
        if verbose and check.evidence:
            lines.append(f"       evidence: {json.dumps(check.evidence, ensure_ascii=False)}")
    counts = report.counts()
    lines.append("-" * 78)
    lines.append(
        "汇总: "
        + ", ".join(f"{k}={counts.get(k, 0)}" for k in (PASS, WARN, FAIL, SKIP))
    )
    verdict = {0: "通过", 1: "不通过", 2: "结论不足"}.get(report.exit_code(), "未知")
    lines.append(f"结论: {verdict}（退出码 {report.exit_code()}）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"错误: {root} 不是目录", file=sys.stderr)
        return 3
    excludes = tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDES
    baseline = build_baseline(root, excludes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"已写入基线: {out}（{len(baseline['files'])} 个文件，"
        f"排除 {len(excludes)} 条规则）"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    report = Report()

    if not root.is_dir():
        print(f"错误: {root} 不是目录", file=sys.stderr)
        return 3

    core = CoreAPI()

    check_python(report)
    check_arch(report)
    check_filesystem(report, root, args.allow_container)
    check_uid_map(report)
    check_bundle(report, root, core)
    check_permissions(report, root)
    check_native_modules(report, root, core)

    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = load_baseline(baseline_path)
        if baseline is None:
            report.add(
                "deploy.integrity",
                FAIL,
                f"基线不可用或格式不符: {baseline_path}",
                "用 `record` 在出厂机上重新采集，或检查文件是否被截断。",
            )
        else:
            check_digests(report, root, baseline, args.strict)
    else:
        report.add(
            "deploy.integrity",
            SKIP,
            "未提供 --baseline",
            "验收需要基线才能判定文件是否被改动；请先用 `record` 采集。",
        )

    check_license(
        report,
        core,
        Path(args.license) if args.license else None,
        args.machine_key,
        args.cloud,
    )
    check_runtime(report, core, args.runtime_probe, args.worker_timeout)

    text = render_text(report, args.verbose)
    print(text)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["target"] = {
            "root": str(root),
            "license": str(Path(args.license)) if args.license else None,
            "baseline": str(Path(args.baseline)) if args.baseline else None,
            "cloud": bool(args.cloud),
            "strict": bool(args.strict),
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n报告已写入: {out}")

    return report.exit_code()


def cmd_doctor(args: argparse.Namespace) -> int:
    """无需许可证的环境自检。"""
    root = Path(args.root).resolve()
    report = Report()
    core = CoreAPI()
    check_python(report)
    check_arch(report)
    check_filesystem(report, root, args.allow_container)
    check_uid_map(report)
    check_bundle(report, root, core)
    if core.load():
        report.add("runtime.import_core", PASS, "_core 导入成功")
    else:
        detail, remedy = core.explain_import_error()
        report.add("runtime.import_core", FAIL, detail, remedy)
    print(render_text(report, args.verbose))
    return report.exit_code()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Jasna 离线部署与现场验收校验器（只读，不解密任何权重）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "必须在冻结发行版进程内运行（Nuitka 构建没有独立解释器），\n"
            "否则 /proc/self/exe 指向外部进程，_core 会拒绝加载。\n"
            "配套的 .enc 清单校验见 jasna_bundle_check.py（无需运行时）。"
        ),
    )
    parser.add_argument("--version", action="version", version=f"{TOOL} {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default=".", help="部署根目录（默认当前目录）")
        p.add_argument("--verbose", "-v", action="store_true", help="输出证据明细")
        p.add_argument(
            "--allow-container",
            action="store_true",
            help="允许 overlay 等容器文件系统（降级为 WARN）",
        )

    p_record = sub.add_parser("record", help="采集部署基线（出厂机执行一次）")
    common(p_record)
    p_record.add_argument("--out", required=True, help="基线输出路径")
    p_record.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="额外排除的 glob 规则，可重复；提供后覆盖默认排除表",
    )
    p_record.set_defaults(func=cmd_record)

    p_verify = sub.add_parser("verify", help="现场验收校验")
    common(p_verify)
    p_verify.add_argument("--license", help="许可证文件路径")
    p_verify.add_argument("--baseline", help="基线 JSON 路径")
    p_verify.add_argument(
        "--machine-key",
        help="覆盖 product 返回的机器码（当你的发行版采用不同编码时使用）",
    )
    p_verify.add_argument(
        "--cloud", action="store_true", help="按云端工作区模式取机器码"
    )
    p_verify.add_argument(
        "--strict", action="store_true", help="基线外多出的文件视为 FAIL"
    )
    p_verify.add_argument(
        "--runtime-probe",
        action="store_true",
        help="启动一次 LTX worker 以验证准入握手（会短暂创建子进程）",
    )
    p_verify.add_argument(
        "--worker-timeout",
        type=int,
        default=30,
        help="worker 探针的说明性超时（秒），默认 30",
    )
    p_verify.add_argument("--report", help="验收报告 JSON 输出路径")
    p_verify.set_defaults(func=cmd_verify)

    p_doctor = sub.add_parser("doctor", help="无需许可证的环境自检")
    common(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"内部错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
