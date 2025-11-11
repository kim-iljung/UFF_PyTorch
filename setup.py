from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup

try:
    import rdkit  # type: ignore
    from rdkit import RDConfig  # type: ignore
except ImportError as exc:  # pragma: no cover - build-time guard
    raise RuntimeError(
        "RDKit must be installed before building uff-torch extensions"
    ) from exc


def _candidate_lib_dirs() -> list[Path]:
    base = Path(rdkit.__file__).resolve().parent
    candidates = {
        base.parent / "rdkit.libs",
        base / "rdkit.libs",
        Path(os.environ.get("RDKIT_LIBRARY_DIR", "")),
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/local/lib"),
    }
    return [path for path in candidates if path and path.is_dir()]


def _find_library(name: str, lib_dirs: list[Path]) -> tuple[str, Path]:
    for lib_dir in lib_dirs:
        direct = list(lib_dir.glob(f"lib{name}.so"))
        if direct:
            return name, lib_dir
        matches = list(lib_dir.glob(f"lib{name}*.so*"))
        if matches:
            match = sorted(matches)[0]
            return f":{match.name}", lib_dir
    raise RuntimeError(f"Could not locate the RDKit library '{name}'")


def _resolved(path: Path) -> Path:
    """Best-effort resolve of *path* without failing."""

    try:
        return path.resolve()
    except OSError:
        return path


def _rdkit_include_dirs() -> list[str]:
    include_dirs: list[str] = []
    seen: set[str] = set()

    def add_with_variants(base: Path) -> None:
        if not base:
            return
        candidates = [
            base,
            base / "include",
            base / "include" / "rdkit",
            base / "Library" / "include",
            base / "Library" / "include" / "rdkit",
            base / "rdkit",
            base / "Code",
            base / "External",
            base / "share" / "RDKit",
            base / "share" / "RDKit" / "Code",
        ]
        for candidate in candidates:
            _maybe_add(candidate, include_dirs, seen)

    include_env = os.environ.get("RDKIT_INCLUDE_DIR")
    if include_env:
        for entry in include_env.split(os.pathsep):
            if entry:
                _maybe_add(Path(entry), include_dirs, seen)

    rdmodule_path = _resolved(Path(rdkit.__file__)).parent
    add_with_variants(rdmodule_path)
    add_with_variants(rdmodule_path.parent)

    rdconfig_candidates = [
        Path(getattr(RDConfig, "RDBaseDir", "")),
        Path(getattr(RDConfig, "RDCodeDir", "")),
        Path(getattr(RDConfig, "RDIncDir", "")),
        Path(getattr(RDConfig, "RDBoostDir", "")),
    ]
    for candidate in rdconfig_candidates:
        add_with_variants(candidate)

    prefix_values = [
        sys.prefix,
        sys.exec_prefix,
        getattr(sys, "base_prefix", ""),
        getattr(sys, "base_exec_prefix", ""),
        os.environ.get("CONDA_PREFIX", ""),
        os.environ.get("VIRTUAL_ENV", ""),
    ]
    prefixes = {Path(value) for value in prefix_values if value}
    for prefix in prefixes:
        add_with_variants(prefix)

    system_candidates = [
        Path("/usr/include/rdkit"),
        Path("/usr/local/include/rdkit"),
        Path("/opt/homebrew/include/rdkit"),
    ]
    for candidate in system_candidates:
        _maybe_add(candidate, include_dirs, seen)

    target_header = Path("GraphMol/ForceFieldHelpers/UFF/AtomTyper.h")
    for directory in list(include_dirs):
        path = Path(directory)
        if (path / target_header).is_file():
            break
    else:
        search_space = [Path(dir_path) for dir_path in include_dirs]
        for base in list(search_space):
            if not base.is_dir():
                continue
            candidate = base / "rdkit"
            if (candidate / target_header).is_file():
                _maybe_add(candidate, include_dirs, seen)
                break
            candidate = base / "Code"
            if (candidate / target_header).is_file():
                _maybe_add(candidate, include_dirs, seen)
                break
        else:
            raise RuntimeError(
                "Could not determine RDKit include directories containing"
                " GraphMol/ForceFieldHelpers/UFF/AtomTyper.h."
                " Set RDKIT_INCLUDE_DIR to the path exposing RDKit headers."
            )

    return include_dirs


def _maybe_add(path: Path, dest: list[str], seen: set[str]) -> None:
    if not path:
        return
    path = _resolved(path)
    if path.is_dir():
        resolved = str(path)
        if resolved not in seen:
            dest.append(resolved)
            seen.add(resolved)


def _parse_env_paths(*names: str) -> list[Path]:
    """Return directories from environment variables that may hold Boost headers."""

    paths: list[Path] = []
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        for entry in raw.split(os.pathsep):
            if not entry:
                continue
            candidate = Path(entry)
            if candidate.is_dir():
                paths.append(candidate)
    return paths


def _yield_boost_roots(base: Path) -> list[Path]:
    """Collect plausible include roots beneath *base* containing boost/python.hpp."""

    candidates: list[Path] = []

    def add_if_header(root: Path) -> None:
        header = root / "boost" / "python.hpp"
        if header.is_file():
            candidates.append(root)

    add_if_header(base)

    for pattern in ("boost", "boost-*", "boost_*", "boost.*"):
        for candidate in base.glob(pattern):
            if not candidate.is_dir():
                continue
            if candidate.name == "boost":
                add_if_header(candidate.parent)
                continue
            if (candidate / "boost" / "python.hpp").is_file():
                candidates.append(candidate)

    direct = base / "python.hpp"
    if direct.is_file() and base.name == "boost":
        parent = base.parent
        if parent.is_dir():
            candidates.append(parent)

    return candidates


def _boost_include_dirs(existing: list[str], lib_dirs: list[Path]) -> list[str]:
    """Augment *existing* include directories with Boost headers if needed."""

    def has_boost_header(directory: str) -> bool:
        return (Path(directory) / "boost" / "python.hpp").is_file()

    if any(has_boost_header(path) for path in existing):
        return existing

    include_dirs = list(existing)
    seen: set[str] = set(include_dirs)

    for env_path in _parse_env_paths(
        "BOOST_INCLUDEDIR",
        "BOOST_ROOT",
        "BOOSTROOT",
        "BOOST_HOME",
    ):
        _maybe_add(env_path, include_dirs, seen)
        _maybe_add(env_path / "include", include_dirs, seen)

    rdconfig_candidates = [
        Path(getattr(RDConfig, "boostIncludeDir", "")),
        Path(getattr(RDConfig, "RDBoostDir", "")),
    ]
    for candidate in rdconfig_candidates:
        _maybe_add(candidate, include_dirs, seen)

    prefix_candidates = {
        Path(sys.prefix),
        Path(sys.exec_prefix),
        Path(getattr(sys, "base_prefix", "")),
        Path(getattr(sys, "base_exec_prefix", "")),
        Path(os.environ.get("CONDA_PREFIX", "")),
        Path(os.environ.get("VIRTUAL_ENV", "")),
    }
    for prefix in prefix_candidates:
        if not prefix:
            continue
        _maybe_add(prefix / "include", include_dirs, seen)

    for lib_dir in lib_dirs:
        _maybe_add(lib_dir.parent / "include", include_dirs, seen)
        _maybe_add(lib_dir.parent / "include" / "boost", include_dirs, seen)

    if not any(has_boost_header(path) for path in include_dirs):
        expanded = list(include_dirs)
        for entry in expanded:
            base = Path(entry)
            if not base.is_dir():
                continue
            for candidate in _yield_boost_roots(base):
                _maybe_add(candidate, include_dirs, seen)

    if not any(has_boost_header(path) for path in include_dirs):
        raise RuntimeError(
            "Could not locate Boost headers (missing boost/python.hpp)."
            " Set BOOST_INCLUDEDIR or BOOST_ROOT explicitly."
        )

    return include_dirs


required_libraries = [
    "RDKitForceFieldHelpers",
    "RDKitForceField",
    "RDKitGenericGroups",
    "RDKitSmilesParse",
    "RDKitSubstructMatch",
    "RDKitGraphMol",
    "RDKitRDGeometryLib",
    "RDKitDataStructs",
    "RDKitRDGeneral",
    "RDKitRDBoost",
    "boost_python311",
]

lib_dirs = _candidate_lib_dirs()
if not lib_dirs:
    raise RuntimeError("Could not locate RDKit library directories")

libraries: list[str] = []
runtime_dirs: set[str] = set()
for lib in required_libraries:
    resolved, directory = _find_library(lib, lib_dirs)
    libraries.append(resolved)
    runtime_dirs.add(str(directory))

python_include = sysconfig.get_paths()["include"]
rdkit_includes = _rdkit_include_dirs()
if not rdkit_includes:
    raise RuntimeError(
        "Could not determine RDKit include directories."
        " Set RDKIT_INCLUDE_DIR explicitly."
    )
include_dirs = _boost_include_dirs([python_include, *rdkit_includes], lib_dirs)

ext_modules = [
    Extension(
        "uff_torch._atom_typing",
        sources=["uff_torch/_atom_typing.cpp"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=list(runtime_dirs),
        runtime_library_dirs=list(runtime_dirs),
        extra_compile_args=["-std=c++17"],
        language="c++",
    )
]

setup(ext_modules=ext_modules)
