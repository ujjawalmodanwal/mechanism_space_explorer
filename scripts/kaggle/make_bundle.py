"""
Create deployment bundles for running nanoreactor refinement on Kaggle GPUs.

Packages repository source code, system definitions, candidate files, seed xTB
reaction data, and device-configured engine configurations into a zip bundle.
"""

import argparse
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "add_bundle_arguments",
    "build_bundle",
    "build_parser",
    "check_engine_model_starts_with_models",
    "collect_base_bundle_files",
    "collect_bundle_files",
    "collect_directory_files",
    "collect_optional_model_files",
    "collect_seed_files",
    "collect_split_files",
    "collect_system_definition_files",
    "collect_system_files",
    "create_bundle_metadata",
    "format_device_cuda_line",
    "format_enable_cueq_line",
    "format_yaml_key_line",
    "get_git_commit",
    "get_git_dirty",
    "get_git_info",
    "is_top_level_section",
    "load_engines_config",
    "main",
    "transform_engine_cueq_lines",
    "transform_engine_device_lines",
    "transform_engine_setting_lines",
    "update_engine_cueq",
    "update_engine_device",
    "write_bundle_zip",
]

EXCLUDED_NAMES = {
    ".DS_Store",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def is_top_level_section(line: str) -> bool:
    """
    Check whether a YAML line starts an unindented top-level section.

    :param line: Raw line from YAML file.
    :type line: str
    :return: True if line is a non-comment unindented key.
    :rtype: bool
    """
    stripped = line.strip()
    return (
        bool(stripped)
        and not line.startswith((" ", "\t"))
        and not stripped.startswith("#")
    )


def format_yaml_key_line(line: str, key: str, value: str) -> str:
    """
    Format replacement YAML key line preserving indent and trailing comment.

    :param line: Original key line from YAML file.
    :type line: str
    :param key: YAML setting key.
    :type key: str
    :param value: New setting value string.
    :type value: str
    :return: Formatted replacement line.
    :rtype: str
    """
    indent = len(line) - len(line.lstrip())
    comment_idx = line.find("#")
    if comment_idx != -1:
        comment = line[comment_idx:].strip()
        return f"{' ' * indent}{key}: {value}  {comment}\n"
    return f"{' ' * indent}{key}: {value}\n"


def format_device_cuda_line(line: str) -> str:
    """
    Format replacement device line preserving indent and trailing comment.

    :param line: Original device line from YAML file.
    :type line: str
    :return: Formatted line setting device to cuda.
    :rtype: str
    """
    return format_yaml_key_line(line, "device", "cuda")


def format_enable_cueq_line(line: str) -> str:
    """
    Format replacement enable_cueq line preserving indent and trailing comment.

    :param line: Original enable_cueq line from YAML file.
    :type line: str
    :return: Formatted line setting enable_cueq to true.
    :rtype: str
    """
    return format_yaml_key_line(line, "enable_cueq", "true")


def transform_engine_setting_lines(
    lines: Sequence[str],
    target_engine: str,
    key: str,
    format_line: Callable[[str], str],
) -> tuple[list[str], bool, bool]:
    """
    Transform lines matching key within target engine and track matches.

    :param lines: Sequence of raw text lines from engines.yaml.
    :type lines: Sequence[str]
    :param target_engine: Target engine identifier to modify.
    :type target_engine: str
    :param key: Configuration key prefix to match.
    :type key: str
    :param format_line: Formatting function for replacement line.
    :type format_line: Callable[[str], str]
    :return: Tuple of transformed lines, engine found flag, key found flag.
    :rtype: tuple[list[str], bool, bool]
    """
    in_eng, eng_found, key_found = False, False, False
    out: list[str] = []
    prefix = f"{key}:"
    for line in lines:
        if is_top_level_section(line):
            in_eng = line.strip().startswith(f"{target_engine}:")
            eng_found = eng_found or in_eng
        if in_eng and line.strip().startswith(prefix):
            out.append(format_line(line))
            key_found = True
        else:
            out.append(line)
    return out, eng_found, key_found


def transform_engine_device_lines(
    lines: Sequence[str],
    target_engine: str,
) -> tuple[list[str], bool, bool]:
    """
    Transform lines setting target engine device to cuda and track matches.

    :param lines: Sequence of raw text lines from engines.yaml.
    :type lines: Sequence[str]
    :param target_engine: Target engine identifier to modify.
    :type target_engine: str
    :return: Tuple of transformed lines, engine found flag, device found flag.
    :rtype: tuple[list[str], bool, bool]
    """
    return transform_engine_setting_lines(
        lines, target_engine, "device", format_device_cuda_line
    )


def transform_engine_cueq_lines(
    lines: Sequence[str],
    target_engine: str = "mace",
) -> tuple[list[str], bool, bool]:
    """
    Transform lines setting enable_cueq to true for target engine.

    :param lines: Sequence of raw text lines from engines.yaml.
    :type lines: Sequence[str]
    :param target_engine: Target engine identifier to modify.
    :type target_engine: str
    :return: Tuple of transformed lines, engine found flag, key found flag.
    :rtype: tuple[list[str], bool, bool]
    """
    return transform_engine_setting_lines(
        lines, target_engine, "enable_cueq", format_enable_cueq_line
    )


def update_engine_device(
    lines: Sequence[str],
    target_engine: str,
) -> str:
    """
    Update device line to cuda for target engine while preserving comments.

    :param lines: Sequence of raw text lines from engines.yaml.
    :type lines: Sequence[str]
    :param target_engine: Target engine identifier to modify.
    :type target_engine: str
    :return: Transformed YAML content string.
    :rtype: str
    :raises ValueError: If target engine or device line is not found.
    """
    out, eng_found, dev_found = transform_engine_device_lines(
        lines, target_engine
    )
    if not eng_found:
        raise ValueError(f"Engine '{target_engine}' not found in engines.yaml")
    if not dev_found:
        raise ValueError(
            f"device: setting not found for engine '{target_engine}'"
        )
    return "".join(out)


def update_engine_cueq(
    lines: Sequence[str],
    target_engine: str = "mace",
) -> str:
    """
    Update enable_cueq to true for target engine while preserving comments.

    :param lines: Sequence of raw text lines from engines.yaml.
    :type lines: Sequence[str]
    :param target_engine: Target engine identifier to modify.
    :type target_engine: str
    :return: Transformed YAML content string.
    :rtype: str
    :raises ValueError: If target engine or enable_cueq line is not found.
    """
    out, eng_found, key_found = transform_engine_cueq_lines(
        lines, target_engine
    )
    if not eng_found:
        raise ValueError(f"Engine '{target_engine}' not found in engines.yaml")
    if not key_found:
        raise ValueError(
            f"enable_cueq: setting not found for engine '{target_engine}'"
        )
    return "".join(out)


def check_engine_model_starts_with_models(
    lines: Sequence[str],
    target_engine: str,
) -> bool:
    """
    Check if the target engine's model path begins with 'models/'.

    :param lines: Sequence of raw text lines from engines.yaml.
    :type lines: Sequence[str]
    :param target_engine: Target engine identifier to inspect.
    :type target_engine: str
    :return: True if model path starts with 'models/', False otherwise.
    :rtype: bool
    """
    in_engine = False
    for line in lines:
        if is_top_level_section(line):
            in_engine = line.strip().startswith(f"{target_engine}:")
        if in_engine and line.strip().startswith("model:"):
            val = line.strip()[len("model:") :].strip().strip("\"'")
            return val.startswith("models/")
    return False


def get_git_commit(repo_dir: Path) -> str:
    """
    Retrieve short Git commit hash for repository.

    :param repo_dir: Path to repository root directory.
    :type repo_dir: Path
    :return: Short commit hash or 'uncommitted'.
    :rtype: str
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return "uncommitted"


def get_git_dirty(repo_dir: Path) -> bool:
    """
    Determine whether repository working tree has uncommitted modifications.

    :param repo_dir: Path to repository root directory.
    :type repo_dir: Path
    :return: True if dirty status output is present.
    :rtype: bool
    """
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def get_git_info(repo_dir: Path) -> tuple[str, bool]:
    """
    Retrieve short Git commit hash and dirty status for repository.

    :param repo_dir: Path to repository root directory.
    :type repo_dir: Path
    :return: Tuple of short commit hash and boolean dirty indicator.
    :rtype: tuple[str, bool]
    """
    return get_git_commit(repo_dir), get_git_dirty(repo_dir)


def collect_directory_files(
    directory: Path,
    repo_root: Path,
) -> list[tuple[Path, str]]:
    """
    Recursively collect files from a directory, excluding unwanted names.

    :param directory: Directory to inspect.
    :type directory: Path
    :param repo_root: Base repository root for relative paths.
    :type repo_root: Path
    :return: List of tuples containing source file path and relative path.
    :rtype: list[tuple[Path, str]]
    """
    collected: list[tuple[Path, str]] = []
    if not directory.exists():
        return collected
    for path in sorted(directory.rglob("*")):
        if (
            path.is_file()
            and not set(path.parts).intersection(EXCLUDED_NAMES)
            and path.suffix not in {".pyc", ".pyo"}
        ):
            collected.append((path, path.relative_to(repo_root).as_posix()))
    return collected


def collect_base_bundle_files(
    repo_root: Path,
) -> list[tuple[Path, str]]:
    """
    Collect source tree, search config, and run script for deployment bundle.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :return: List of file path pairs for core bundle components.
    :rtype: list[tuple[Path, str]]
    :raises FileNotFoundError: If required files or directories are missing.
    """
    src_dir = repo_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing src directory at {src_dir}")
    search_yaml = repo_root / "configs" / "search.yaml"
    if not search_yaml.exists():
        raise FileNotFoundError(f"Missing search config at {search_yaml}")
    runner = repo_root / "scripts" / "kaggle" / "run_refine.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing runner script at {runner}")
    entries = collect_directory_files(src_dir, repo_root)
    entries.append((search_yaml, "configs/search.yaml"))
    entries.append((runner, "scripts/kaggle/run_refine.py"))
    return entries


def collect_seed_files(
    repo_root: Path,
    system: str,
) -> list[tuple[Path, str]]:
    """
    Collect seed xTB calculation files ensuring reactions.jsonl exists.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param system: System identifier string.
    :type system: str
    :return: List of file path pairs for seed results.
    :rtype: list[tuple[Path, str]]
    :raises FileNotFoundError: If reactions.jsonl is missing.
    """
    seed_dir = repo_root / "results" / "refine" / system / "xtb"
    seed_rxns = seed_dir / "reactions.jsonl"
    if not seed_rxns.exists():
        raise FileNotFoundError(f"Missing seed reactions for '{system}'")
    return collect_directory_files(seed_dir, repo_root)


def collect_system_definition_files(
    repo_root: Path,
    system: str,
) -> list[tuple[Path, str]]:
    """
    Collect candidate inputs and system configuration files.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param system: System identifier string.
    :type system: str
    :return: List of file path pairs for candidates and systems.
    :rtype: list[tuple[Path, str]]
    :raises FileNotFoundError: If candidates or systems directory is missing.
    """
    cand_dir = repo_root / "candidates" / system
    if not cand_dir.exists():
        raise FileNotFoundError(f"Missing candidates: {cand_dir}")
    sys_dir = repo_root / "systems" / system
    if not sys_dir.exists():
        raise FileNotFoundError(f"Missing system: {sys_dir}")
    return collect_directory_files(
        cand_dir, repo_root
    ) + collect_directory_files(sys_dir, repo_root)


def collect_system_files(
    repo_root: Path,
    system: str,
) -> list[tuple[Path, str]]:
    """
    Collect candidates, system files, and seed xTB reactions for one system.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param system: System identifier string.
    :type system: str
    :return: List of file path pairs for the specified system.
    :rtype: list[tuple[Path, str]]
    :raises FileNotFoundError: If required directories or seed reactions fail.
    """
    return collect_system_definition_files(
        repo_root, system
    ) + collect_seed_files(repo_root, system)


def collect_optional_model_files(
    repo_root: Path,
    include_models: bool,
) -> list[tuple[Path, str]]:
    """
    Collect model artifacts if requested by engine configuration.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param include_models: Whether model directory should be bundled.
    :type include_models: bool
    :return: List of file path pairs for model directory.
    :rtype: list[tuple[Path, str]]
    :raises FileNotFoundError: If models directory is required but missing.
    """
    if not include_models:
        return []
    models_dir = repo_root / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models directory at {models_dir}")
    return collect_directory_files(models_dir, repo_root)


def collect_split_files(
    split_paths: Sequence[Path] | None,
    systems: Sequence[str],
) -> list[tuple[Path, str]]:
    """
    Collect reaction split JSON files and validate against target systems.

    :param split_paths: Optional sequence of paths to split JSON files.
    :type split_paths: Sequence[Path] | None
    :param systems: Sequence of allowed system identifiers.
    :type systems: Sequence[str]
    :return: List of file path pairs for split configurations.
    :rtype: list[tuple[Path, str]]
    :raises ValueError: If split system is not in target systems.
    """
    if not split_paths:
        return []
    entries: list[tuple[Path, str]] = []
    allowed = set(systems)
    for path in split_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        system = data.get("system")
        if system not in allowed:
            raise ValueError(
                f"Split file '{path}' system '{system}' not in {systems}"
            )
        entries.append((path, f"splits/{system}.json"))
    return entries


def collect_bundle_files(
    repo_root: Path,
    engine: str,
    systems: Sequence[str],
    include_models: bool,
    split_paths: Sequence[Path] | None = None,
) -> list[tuple[Path, str]]:
    """
    Collect all standard disk files to include in deployment bundle.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers to package.
    :type systems: Sequence[str]
    :param include_models: Flag indicating whether models/ should be bundled.
    :type include_models: bool
    :param split_paths: Optional sequence of split JSON file paths.
    :type split_paths: Sequence[Path] | None
    :return: List of file path pairs (source path, archive relative path).
    :rtype: list[tuple[Path, str]]
    :raises FileNotFoundError: If required directories or files are missing.
    :raises ValueError: If split system is not in target systems.
    """
    entries = collect_base_bundle_files(repo_root)
    for system in systems:
        entries.extend(collect_system_files(repo_root, system))
    entries.extend(collect_optional_model_files(repo_root, include_models))
    entries.extend(collect_split_files(split_paths, systems))
    return entries


def create_bundle_metadata(
    repo_root: Path,
    engine: str,
    systems: Sequence[str],
    total_files: int,
) -> bytes:
    """
    Generate formatted JSON bytes for bundle manifest metadata.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param total_files: Total count of files in bundle archive.
    :type total_files: int
    :return: Encoded UTF-8 JSON bytes.
    :rtype: bytes
    """
    commit, dirty = get_git_info(repo_root)
    metadata = {
        "engine": engine,
        "systems": list(systems),
        "git_commit": commit,
        "dirty": dirty,
        "utc_time": datetime.now(UTC).isoformat(),
        "file_count": total_files,
    }
    return json.dumps(metadata, indent=2).encode("utf-8")


def write_bundle_zip(
    out_path: Path,
    disk_files: Sequence[tuple[Path, str]],
    engines_yaml: str,
    metadata_bytes: bytes,
) -> None:
    """
    Write collected files and in-memory assets into destination zip archive.

    :param out_path: Destination zip archive file path.
    :type out_path: Path
    :param disk_files: Sequence of source and archive path pairs.
    :type disk_files: Sequence[tuple[Path, str]]
    :param engines_yaml: Modified engines.yaml text content.
    :type engines_yaml: str
    :param metadata_bytes: Encoded JSON metadata bytes.
    :type metadata_bytes: bytes
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as arc:
        for file_path, archive_name in disk_files:
            arc.write(file_path, arcname=archive_name)
        arc.writestr("configs/engines.yaml", engines_yaml)
        arc.writestr("BUNDLE.json", metadata_bytes)


def load_engines_config(
    repo_root: Path,
    engine: str,
    enable_cueq: bool = False,
) -> tuple[str, bool]:
    """
    Load engines.yaml, configure target engine device, and check model flag.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param enable_cueq: Whether to enable cuEquivariance kernels for mace.
    :type enable_cueq: bool
    :return: Tuple of modified engines YAML text and include_models flag.
    :rtype: tuple[str, bool]
    :raises FileNotFoundError: If configs/engines.yaml does not exist.
    """
    path = repo_root / "configs" / "engines.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing engines config at {path}")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    text = update_engine_device(lines, engine)
    if enable_cueq:
        text = update_engine_cueq(
            text.splitlines(keepends=True), target_engine="mace"
        )
    return text, (
        check_engine_model_starts_with_models(lines, engine)
    )


def build_bundle(
    repo_root: Path,
    engine: str,
    systems: Sequence[str],
    out_path: Path,
    split_paths: Sequence[Path] | None = None,
    enable_cueq: bool = False,
) -> tuple[int, int]:
    """
    Construct zip bundle with configured files and bundle metadata.

    :param repo_root: Base repository root path.
    :type repo_root: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers to package.
    :type systems: Sequence[str]
    :param out_path: Destination zip archive file path.
    :type out_path: Path
    :param split_paths: Optional sequence of split JSON file paths.
    :type split_paths: Sequence[Path] | None
    :param enable_cueq: Whether to enable cuEquivariance kernels for mace.
    :type enable_cueq: bool
    :return: Tuple of total file count and archive size in bytes.
    :rtype: tuple[int, int]
    """
    engines_yaml, include_models = load_engines_config(
        repo_root, engine, enable_cueq=enable_cueq
    )
    disk_files = collect_bundle_files(
        repo_root, engine, systems, include_models, split_paths
    )
    total_files = len(disk_files) + 2
    metadata = create_bundle_metadata(repo_root, engine, systems, total_files)
    write_bundle_zip(out_path, disk_files, engines_yaml, metadata)
    return total_files, out_path.stat().st_size


def add_bundle_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register command line arguments on make_bundle parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--engine", required=True, help="Engine identifier (e.g. mace)."
    )
    parser.add_argument(
        "--systems", nargs="+", required=True, help="System identifiers."
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output destination zip path."
    )
    parser.add_argument(
        "--split-file",
        dest="split_files",
        action="append",
        type=Path,
        default=None,
        help="Optional path to a split JSON file (repeatable).",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        default=False,
        help="Enable NVIDIA cuEquivariance kernels in bundled MACE config.",
    )
    parser.add_argument(
        "--repo-root", default=Path("."), type=Path, help="Repository root."
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for make_bundle CLI.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Build self-contained Kaggle refinement bundle."
    )
    add_bundle_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute make_bundle command line interface.

    :param argv: Optional command-line arguments.
    :type argv: Sequence[str] | None
    :return: Process exit status code.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    count, size = build_bundle(
        repo_root=args.repo_root.resolve(),
        engine=args.engine,
        systems=args.systems,
        out_path=args.out.resolve(),
        split_paths=args.split_files,
        enable_cueq=args.enable_cueq,
    )
    mb = size / (1024 * 1024)
    print(
        f"Bundled {count} files for engine '{args.engine}' "
        f"into {args.out} ({mb:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
