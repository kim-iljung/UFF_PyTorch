from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup

try:
    import rdkit  # type: ignore
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
    include_env = os.environ.get("RDKIT_INCLUDE_DIR")
    include_dirs: list[str] = []
    if include_env:
        path = Path(include_env)
        if path.is_dir():
            include_dirs.append(str(path))
    wheel_candidate = Path(rdkit.__file__).resolve().parent / "include"
    if wheel_candidate.is_dir():
        include_dirs.append(str(wheel_candidate))
    system_candidate = Path("/usr/include/rdkit")
    if system_candidate.is_dir():
        include_dirs.append(str(system_candidate))
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
