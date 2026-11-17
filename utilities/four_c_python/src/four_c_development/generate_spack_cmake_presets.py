# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Generate CMake presets that match a concretized checkout-local Spack environment."""

import json
from pathlib import Path
import shlex
import subprocess
import sys


VARIANT_CACHE_VARIABLES = {
    "cuda": "FOUR_C_CLANGCUDA",
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
    if parameters.get("cuda") is True:
        cache_variables["FOUR_C_ENABLE_LINKER_DETECTION"] = "OFF"
    return cache_variables


def get_compiler_configuration(
    environment_directory: Path, *, cuda: bool
) -> tuple[dict[str, str], dict[str, str]]:
    """Query the root's compilers and preserve the recipe's CUDA build environment."""

    cuda_variables = (
        ("CLANGCUDA_ARCH", "CLANGCUDA_CLANG_PATH", "CLANGCUDA_CUDA_PATH")
        if cuda
        else ()
    )
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
            f"name: os.environ[name] for name in {('SPACK_CC', 'SPACK_CXX') + cuda_variables!r}"
            "}))",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    build_environment = json.loads(result.stdout)
    environment = {name: build_environment[name] for name in cuda_variables}
    cache_variables = {
        "CMAKE_C_COMPILER": build_environment["SPACK_CC"],
        "CMAKE_CXX_COMPILER": build_environment["SPACK_CXX"],
        "MPI_C_COMPILER": "${sourceDir}/.spack/.spack-env/view/bin/mpicc",
        "MPI_CXX_COMPILER": "${sourceDir}/.spack/.spack-env/view/bin/mpicxx",
    }
    if cuda:
        # Bake the settings into the compiler so plain Ninja builds do not need exports.
        compiler = environment_directory / "clangcuda++"
        wrapper = environment_directory.parent / "utilities/clangcuda++"
        compiler.write_text(
            "#!/bin/sh\n"
            + "".join(
                f"export {name}={shlex.quote(value)}\n"
                for name, value in environment.items()
            )
            + f'exec {shlex.quote(str(wrapper))} "$@"\n',
            encoding="utf-8",
        )
        compiler.chmod(0o755)
        cache_variables["CMAKE_CXX_COMPILER"] = "${sourceDir}/.spack/clangcuda++"
    return cache_variables, environment


def get_generated_presets(
    cache_variables: dict[str, str], environment: dict[str, str]
) -> dict[str, object]:
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
                    **environment,
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
    compiler_variables, environment = get_compiler_configuration(
        environment_directory, cuda=cache_variables.get("FOUR_C_CLANGCUDA") == "ON"
    )
    cache_variables.update(compiler_variables)
    generated_presets.write_text(
        json.dumps(get_generated_presets(cache_variables, environment), indent=2)
        + "\n",
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
