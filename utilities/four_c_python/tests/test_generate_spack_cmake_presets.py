# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Check generated presets using mocked Spack compiler metadata.

Verify CUDA settings and paths with spaces survive wrapper execution in an empty
environment, while CPU presets retain the original compiler without a wrapper.
No Spack installation or CUDA compilation is performed.
"""

from contextlib import chdir
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from four_c_development import generate_spack_cmake_presets as generator


class TestSpackCMakePresets(unittest.TestCase):
    def generate(self, source, *, cuda, build_environment):
        environment = source / ".spack"
        environment.mkdir()
        (environment / "spack.lock").write_text(
            json.dumps(
                {
                    "roots": [{"hash": "root"}],
                    "concrete_specs": {
                        "root": {
                            "name": "4c-multiphysics",
                            "parameters": {"cuda": cuda, "vtk": False},
                        }
                    },
                }
            )
        )
        with (
            chdir(source),
            patch.object(
                generator.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json.dumps(build_environment)
                ),
            ) as query,
        ):
            generator.main()
        self.assertIn("build-env", query.call_args.args[0])
        return json.loads((environment / "4CSpackPresets.json").read_text())[
            "configurePresets"
        ][0]

    def test_cuda_toolchain_survives_without_preset_environment(self):
        with tempfile.TemporaryDirectory(prefix="fourc presets ") as directory:
            source = Path(directory)
            wrapper = source / "utilities/clangcuda++"
            wrapper.parent.mkdir()
            wrapper.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$CLANGCUDA_ARCH" '
                '"$CLANGCUDA_CLANG_PATH" "$CLANGCUDA_CUDA_PATH" "$@"\n'
            )
            wrapper.chmod(0o755)
            cuda_environment = {
                "CLANGCUDA_ARCH": "sm_90",
                "CLANGCUDA_CLANG_PATH": "/toolchain with spaces/clang++",
                "CLANGCUDA_CUDA_PATH": "/cuda toolkit",
            }
            profile = self.generate(
                source,
                cuda=True,
                build_environment={
                    "SPACK_CC": "/usr/bin/clang",
                    "SPACK_CXX": cuda_environment["CLANGCUDA_CLANG_PATH"],
                    **cuda_environment,
                },
            )
            cache = profile["cacheVariables"]
            self.assertEqual(cache["FOUR_C_CLANGCUDA"], "ON")
            self.assertEqual(cache["FOUR_C_ENABLE_LINKER_DETECTION"], "OFF")
            self.assertEqual(
                cache["CMAKE_CXX_COMPILER"], "${sourceDir}/.spack/clangcuda++"
            )
            self.assertEqual(
                profile["environment"]["OMPI_CXX"], cache["CMAKE_CXX_COMPILER"]
            )
            for name, value in cuda_environment.items():
                self.assertEqual(profile["environment"][name], value)
            result = subprocess.run(
                [str(source / ".spack/clangcuda++"), "argument with spaces"],
                env={},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                [*cuda_environment.values(), "argument with spaces"],
            )

    def test_non_cuda_environment_uses_original_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            profile = self.generate(
                source,
                cuda=False,
                build_environment={
                    "SPACK_CC": "/usr/bin/gcc",
                    "SPACK_CXX": "/usr/bin/g++",
                },
            )
            self.assertEqual(
                profile["cacheVariables"]["CMAKE_CXX_COMPILER"], "/usr/bin/g++"
            )
            self.assertEqual(profile["cacheVariables"]["FOUR_C_CLANGCUDA"], "OFF")
            self.assertNotIn("CLANGCUDA_ARCH", profile["environment"])
            self.assertFalse((source / ".spack/clangcuda++").exists())
