# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import py4C
import pytest


def test_full_constrained_mixture_fiber():
    material = py4C.mixture.ExponentialFiberMaterial(1.0, 1.0, True)
    gnr = py4C.mixture.LinearGrowthPoissonTurnover(1.0, 10.0)
    fiber = py4C.mixture.FullConstrainedMixtureFiber(
        material, gnr, 1.1, py4C.mixture.HistoryAdaptionStrategy.none, True
    )
    fiber.reinitialize_history(1.1, 0.0)
    fiber.recompute_state(1.11, 0.1, 0.1)
    fiber.update()
    fiber.recompute_state(1.12, 0.2, 0.1)

    assert fiber.cauchy_stress == pytest.approx(1.5495817335681483)
    assert fiber.growth_scalar == pytest.approx(1.5019595353576574)
    assert fiber.history_size == 2
    assert fiber.get_history_times() == pytest.approx([0.0, 0.1])
