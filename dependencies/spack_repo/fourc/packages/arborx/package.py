# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.packages.arborx.package import Arborx as BuiltinArborx

from spack.package import *


class Arborx(BuiltinArborx):
    """ArborX with the post-2.1 fixes required by 4C."""

    # Internal version for post-2.1 floating-point exception fixes. Replace with
    # an official release containing them before removing this overlay.
    version("2.1-4c", commit="826d457a53cadfd8385f6f8a4ed241bb19c54bde")
