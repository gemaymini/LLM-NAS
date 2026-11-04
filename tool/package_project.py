"""将项目代码、运行配置和已提取的 NAS-Bench 数据打包成 ZIP。

用法（仅依赖 Python 标准库）：
    python tool/package_project.py
    python tool/package_project.py --dry-run
    python tool/package_project.py --root /path/to/project -o /tmp/project.zip
    python tool/package_project.py --data-pattern '*.csv' --data-pattern '*.npz'

默认包含代码、README、依赖配置和 *.acc_cache.pkl；--data-pattern 可额外
添加提取数据的文件名模式（递归匹配）。排除隐藏目录、虚拟环境、缓存、
日志、原始模型文件、已有压缩包及 .env；保留 .env.example 配置模板。
注意：当前项目的数据加载器仍要求原始 .pth，本压缩包不包含该文件。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fnmatch
import os
from pathlib import Path
import sys
import tempfile
import zipfile


CODE_SUFFIXES = {
    ".py", ".pyi", ".ipynb", ".sh", ".bash", ".zsh", ".r", ".jl",
    ".c", ".h", ".cpp", ".hpp", ".cu", ".js", ".jsx", ".ts", ".tsx",
}
CONFIG_SUFFIXES = {".toml", ".yaml", ".yml", ".ini", ".cfg"}
PROJECT_FILES = {
    "requirements.txt", "uv.lock", "poetry.lock", "pdm.lock", "pipfile",
    "pipfile.lock", "setup.cfg", "makefile", "dockerfile", "license",
    "license.txt", "license.md", ".env.example", ".python-version", ".gitignore",
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}
EXCLUDED_DIRS = {
    "__pycache__", "node_modules", "venv", "env", "build", "dist", "archives",
    "logs", "site-packages",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".log", ".pth", ".pt", ".zip", ".tar", ".gz", ".bz2",
    ".xz", ".7z", ".rar", ".pem", ".key",
}


def collect_files(root: Path, output: Path, data_patterns: list[str]) -> list[Path]:
    """Collect files without following symlinks or consulting .gitignore."""
    selected = []

    def walk_error(error: OSError) -> None:
        raise error

    for directory, dirs, names in os.walk(root, onerror=walk_error):
        base = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if not name.startswith(".") and name not in EXCLUDED_DIRS
            and not name.endswith(".egg-info") and not (base / name).is_symlink()
        )
        for name in sorted(names):
            path = base / name
            lower = name.lower()
            if path.is_symlink() or not path.is_file() or path == output:
                continue
            if lower.startswith(".env") and lower != ".env.example":
                continue
            if name.startswith(".") and lower not in PROJECT_FILES:
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            if (
                path.suffix.lower() in CODE_SUFFIXES | CONFIG_SUFFIXES
                or lower in PROJECT_FILES
                or lower == "readme" or lower.startswith("readme.")
                or fnmatch.fnmatch(lower, "requirements*.txt")
                or any(fnmatch.fnmatch(name, pattern) for pattern in data_patterns)
            ):
                selected.append(path)
    return sorted(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="要打包的项目目录（默认：脚本所在项目）")
    parser.add_argument("-o", "--output", type=Path, help="输出 ZIP 路径（相对当前工作目录）")
    parser.add_argument("--data-pattern", action="append", default=[], metavar="GLOB",
                        help="额外提取数据的文件名模式，可重复；请用引号包裹")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不创建压缩包")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")
    output = (args.output.expanduser().absolute() if args.output else
              root / "archives" / f"{root.name}_{datetime.now():%Y%m%d_%H%M%S_%f}.zip")
    if output.suffix.lower() != ".zip":
        parser.error("输出文件必须使用 .zip 扩展名")
    if output.exists() or output.is_symlink():
        parser.error(f"输出文件已存在，请指定其他路径：{output}")
    temporary = None
    try:
        files = collect_files(root, output.resolve(), ["*.acc_cache.pkl", *args.data_pattern])
        if not files:
            parser.error("未找到可打包的代码或提取数据")
        total = sum(path.stat().st_size for path in files)
        for path in files:
            print(path.relative_to(root).as_posix())
        print(f"共 {len(files)} 个文件，原始大小 {total / 1024 / 1024:.2f} MiB")
        print(f"输出：{output}")
        if args.dry_run:
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        # Build beside the destination, then publish atomically without overwriting.
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as tmp:
            temporary = Path(tmp.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for path in files:
                archive.write(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")
        os.link(temporary, output)
        print(f"打包完成：{output.stat().st_size / 1024 / 1024:.2f} MiB")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"打包失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
