# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Generate CMake presets that match a concretized checkout-local Spack environment."""

import json
from pathlib import Path
import subprocess
import sys


VARIANT_CACHE_VARIABLES = {
    "qhull": "FOUR_C_WITH_QHULL",
    "arborx": "FOUR_C_WITH_ARBORX",
    "backtrace": "FOUR_C_WITH_BACKTRACE",
    "mirco": "FOUR_C_WITH_MIRCO",
    "dealii": "FOUR_C_WITH_DEAL_II",
    "vtk": "FOUR_C_WITH_VTK",
    "gmsh": "FOUR_C_WITH_GMSH",
    "fftw": "FOUR_C_WITH_FFTW",
    "python": "FOUR_C_WITH_PYTHON",
    "pybind11": "FOUR_C_WITH_PYBIND11",
}


def get_4c_parameters(lockfile: Path) -> dict[str, object]:
    """Return the concretized 4C package parameters from a Spack lockfile."""

    with lockfile.open(encoding="utf-8") as file:
        lock_data = json.load(file)

    concrete_specs = lock_data["concrete_specs"]
    for root in lock_data["roots"]:
        concrete_spec = concrete_specs[root["hash"]]
        if concrete_spec["name"] == "4c-multiphysics":
            return concrete_spec["parameters"]

    raise ValueError("The Spack lockfile does not contain a 4c-multiphysics root spec.")


def get_cache_variables(parameters: dict[str, object]) -> dict[str, str]:
    """Translate enabled 4C Spack variants to their matching CMake cache variables."""

    cache_variables = {
        cmake_variable: "ON" if parameters[variant] else "OFF"
        for variant, cmake_variable in VARIANT_CACHE_VARIABLES.items()
        if isinstance(parameters.get(variant), bool)
    }
    if parameters.get("arborx") is True:
        cache_variables["FOUR_C_ARBORX_FIND_INSTALLED"] = "ON"
    if isinstance(parameters.get("pybind11"), bool):
        cache_variables["FOUR_C_ENABLE_PYTHON_BINDINGS"] = (
            "ON" if parameters["pybind11"] else "OFF"
        )
    if parameters.get("python") is False:
        cache_variables["FOUR_C_ENABLE_METADATA_GENERATION"] = "OFF"
    if isinstance(parameters.get("shared"), bool):
        cache_variables["FOUR_C_BUILD_SHARED_LIBS"] = (
            "ON" if parameters["shared"] else "OFF"
        )
    return cache_variables


def get_compiler_cache_variables(environment_directory: Path) -> dict[str, str]:
    """Query the root's compiler executables, not the cached MPI build's compilers."""

    result = subprocess.run(
        [
            "spack",
            "-e",
            str(environment_directory),
            "build-env",
            "4c-multiphysics",
            "--",
            sys.executable,
            "-c",
            "import json, os; print(json.dumps({"
            "'CMAKE_C_COMPILER': os.environ['SPACK_CC'], "
            "'CMAKE_CXX_COMPILER': os.environ['SPACK_CXX']}))",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(result.stdout) | {
        "MPI_C_COMPILER": "${sourceDir}/.spack/.spack-env/view/bin/mpicc",
        "MPI_CXX_COMPILER": "${sourceDir}/.spack/.spack-env/view/bin/mpicxx",
    }


def get_generated_presets(cache_variables: dict[str, str]) -> dict[str, object]:
    """Build the generated CMake preset document."""

    return {
        "version": 5,
        "include": ["../presets/spack/CMakePresets.json"],
        "configurePresets": [
            {
                "name": ".spack-profile",
                "hidden": True,
                "inherits": ".spack_base",
                "cacheVariables": cache_variables,
                "environment": {
                    "OMPI_CC": cache_variables["CMAKE_C_COMPILER"],
                    "OMPI_CXX": cache_variables["CMAKE_CXX_COMPILER"],
                },
            },
            {
                "name": "spack-generated-debug",
                "displayName": "Debug build with Spack dependencies",
                "inherits": ".spack-profile",
                "binaryDir": "${sourceDir}/build/spack-debug",
                "cacheVariables": {"CMAKE_BUILD_TYPE": "DEBUG"},
            },
            {
                "name": "spack-generated-release",
                "displayName": "Release build with Spack dependencies",
                "inherits": ".spack-profile",
                "binaryDir": "${sourceDir}/build/spack-release",
                "cacheVariables": {"CMAKE_BUILD_TYPE": "RELEASE"},
            },
        ],
    }


def create_user_presets_if_absent(user_presets: Path, generated_presets: Path) -> bool:
    """Create a minimal user preset file that includes generated Spack presets."""

    if user_presets.exists():
        return False

    relative_generated_presets = generated_presets.relative_to(user_presets.parent)
    user_presets.write_text(
        json.dumps(
            {"version": 5, "include": [str(relative_generated_presets)]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def main() -> None:
    """Generate matching CMake presets from the active Spack lockfile."""

    environment_directory = Path(".spack").resolve()
    lockfile = environment_directory / "spack.lock"
    generated_presets = environment_directory / "4CSpackPresets.json"
    source_directory = environment_directory.parent
    user_presets = source_directory / "CMakeUserPresets.json"

    if not lockfile.is_file():
        raise FileNotFoundError(f"Spack lockfile not found: {lockfile}")

    cache_variables = get_cache_variables(get_4c_parameters(lockfile))
    cache_variables.update(get_compiler_cache_variables(environment_directory))
    generated_presets.write_text(
        json.dumps(get_generated_presets(cache_variables), indent=2) + "\n",
        encoding="utf-8",
    )

    if create_user_presets_if_absent(user_presets, generated_presets):
        print(f"Created {user_presets} including {generated_presets}.")
    else:
        print(f"Generated {generated_presets}.")
        print(
            f'Add "include": [".spack/{generated_presets.name}"] to {user_presets} if it is not present.'
        )


if __name__ == "__main__":
    main()
