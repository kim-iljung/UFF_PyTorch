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


def _rdkit_include_dirs() -> list[str]:
    include_dirs: list[str] = []
    seen: set[str] = set()

    include_env = os.environ.get("RDKIT_INCLUDE_DIR")
    if include_env:
        for entry in include_env.split(os.pathsep):
            if not entry:
                continue
            path = Path(entry)
            if path.is_dir():
                resolved = str(path)
                if resolved not in seen:
                    include_dirs.append(resolved)
                    seen.add(resolved)

    base = Path(rdkit.__file__).resolve().parent
    wheel_candidates = [
        base / "include",
        base.parent / "include",
        base.parent / "rdkit" / "include",
    ]
    for candidate in wheel_candidates:
        if candidate.is_dir():
            resolved = str(candidate)
            if resolved not in seen:
                include_dirs.append(resolved)
                seen.add(resolved)

    rdconfig_candidates: list[Path] = []
    rd_base = Path(getattr(RDConfig, "RDBaseDir", ""))
    if rd_base.is_dir():
        rdconfig_candidates.extend(
            [
                rd_base,
                rd_base / "Code",
                rd_base / "External",
            ]
        )
    rd_inc = Path(getattr(RDConfig, "RDIncDir", ""))
    if rd_inc.is_dir():
        rdconfig_candidates.append(rd_inc)
    rd_code = Path(getattr(RDConfig, "RDCodeDir", ""))
    if rd_code.is_dir():
        rdconfig_candidates.append(rd_code)

    for candidate in rdconfig_candidates:
        if candidate.is_dir():
            resolved = str(candidate)
            if resolved not in seen:
                include_dirs.append(resolved)
                seen.add(resolved)

    system_candidate = Path("/usr/include/rdkit")
    if system_candidate.is_dir():
        resolved = str(system_candidate)
        if resolved not in seen:
            include_dirs.append(resolved)
            seen.add(resolved)

    if not include_dirs:
        raise RuntimeError(
            "Could not determine RDKit include directories."
            " Set RDKIT_INCLUDE_DIR explicitly."
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
]


def _resolve_boost_python(lib_dirs: list[Path]) -> tuple[str, Path]:
    """Find the boost_python runtime matching the current interpreter."""

    candidates: list[str] = []
    nodot = sysconfig.get_config_var("py_version_nodot")
    if nodot:
        candidates.append(f"boost_python{nodot}")
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    candidates.extend(
        [
            f"boost_python{py_major}{py_minor}",
            f"boost_python{py_major}",
            "boost_python3",
            "boost_python",
        ]
    )

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            return _find_library(name, lib_dirs)
        except RuntimeError:
            continue

    for lib_dir in lib_dirs:
        matches = sorted(lib_dir.glob("libboost_python*.so*"))
        if matches:
            match = matches[0]
            return f":{match.name}", lib_dir

    raise RuntimeError("Could not locate a suitable boost_python library")

lib_dirs = _candidate_lib_dirs()
if not lib_dirs:
    raise RuntimeError("Could not locate RDKit library directories")

libraries: list[str] = []
runtime_dirs: set[str] = set()
for lib in required_libraries:
    resolved, directory = _find_library(lib, lib_dirs)
    libraries.append(resolved)
    runtime_dirs.add(str(directory))

boost_resolved, boost_dir = _resolve_boost_python(lib_dirs)
libraries.append(boost_resolved)
runtime_dirs.add(str(boost_dir))

python_include = sysconfig.get_paths()["include"]
include_dirs = [python_include, * _rdkit_include_dirs()]

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
